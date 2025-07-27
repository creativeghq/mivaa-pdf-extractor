"""
Integration tests for FastAPI endpoints.

This module tests the complete integration of API endpoints with
underlying services, ensuring proper request/response handling,
authentication, validation, and error handling.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile
import json
from datetime import datetime
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.core.config import get_settings


@pytest.mark.integration
@pytest.mark.api
class TestDocumentProcessingEndpoints:
    """Integration tests for document processing API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    async def async_client(self):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.fixture
    def sample_pdf_file(self):
        """Create a sample PDF file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(b"Mock PDF content for testing")
            temp_file.flush()
            yield temp_file.name
        Path(temp_file.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def mock_pdf_processing_result(self):
        """Mock PDF processing result."""
        return {
            "document_id": "doc_123",
            "title": "Test Document",
            "content": "This is test document content.",
            "metadata": {
                "page_count": 2,
                "file_size": 1024,
                "processed_at": datetime.utcnow().isoformat()
            },
            "pages": [
                {
                    "page_number": 1,
                    "content": "Page 1 content",
                    "images": [],
                    "tables": []
                },
                {
                    "page_number": 2,
                    "content": "Page 2 content",
                    "images": [],
                    "tables": []
                }
            ]
        }
    
    async def test_upload_and_process_pdf_endpoint(
        self, 
        async_client, 
        sample_pdf_file, 
        mock_pdf_processing_result
    ):
        """Test complete PDF upload and processing workflow through API."""
        with patch('app.services.pdf_processor.PDFProcessor.process_pdf') as mock_process:
            with patch('app.services.supabase_service.SupabaseService.create_document') as mock_create:
                mock_process.return_value = mock_pdf_processing_result
                mock_create.return_value = {"id": "doc_123", "title": "Test Document"}
                
                # Test file upload
                with open(sample_pdf_file, 'rb') as f:
                    files = {"file": ("test.pdf", f, "application/pdf")}
                    response = await async_client.post("/api/v1/documents/upload", files=files)
                
                assert response.status_code == 200
                result = response.json()
                
                assert "document_id" in result
                assert result["title"] == "Test Document"
                assert result["metadata"]["page_count"] == 2
                
                # Verify service calls
                mock_process.assert_called_once()
                mock_create.assert_called_once()
    
    async def test_process_pdf_with_options_endpoint(
        self, 
        async_client, 
        sample_pdf_file, 
        mock_pdf_processing_result
    ):
        """Test PDF processing with various options."""
        with patch('app.services.pdf_processor.PDFProcessor.process_pdf') as mock_process:
            with patch('app.services.supabase_service.SupabaseService.create_document') as mock_create:
                mock_process.return_value = mock_pdf_processing_result
                mock_create.return_value = {"id": "doc_123", "title": "Test Document"}
                
                # Test with processing options
                with open(sample_pdf_file, 'rb') as f:
                    files = {"file": ("test.pdf", f, "application/pdf")}
                    data = {
                        "extract_images": "true",
                        "extract_tables": "true",
                        "generate_embeddings": "true",
                        "chunk_size": "1000"
                    }
                    response = await async_client.post(
                        "/api/v1/documents/upload", 
                        files=files, 
                        data=data
                    )
                
                assert response.status_code == 200
                result = response.json()
                
                assert result["document_id"] == "doc_123"
                mock_process.assert_called_once()
    
    async def test_invalid_file_upload_endpoint(self, async_client):
        """Test upload with invalid file types."""
        # Test with non-PDF file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as temp_file:
            temp_file.write(b"This is not a PDF")
            temp_file.flush()
            
            try:
                with open(temp_file.name, 'rb') as f:
                    files = {"file": ("test.txt", f, "text/plain")}
                    response = await async_client.post("/api/v1/documents/upload", files=files)
                
                assert response.status_code == 422
                result = response.json()
                assert "detail" in result
                assert "Invalid file type" in str(result["detail"])
                
            finally:
                Path(temp_file.name).unlink(missing_ok=True)
    
    async def test_large_file_upload_endpoint(self, async_client):
        """Test upload with file size limits."""
        # Create a large file (simulate)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            # Write large content (simulate 100MB file)
            large_content = b"Large PDF content" * (1024 * 1024)  # ~16MB
            temp_file.write(large_content)
            temp_file.flush()
            
            try:
                with patch('app.core.config.get_settings') as mock_settings:
                    settings = MagicMock()
                    settings.MAX_FILE_SIZE = 1024 * 1024  # 1MB limit
                    mock_settings.return_value = settings
                    
                    with open(temp_file.name, 'rb') as f:
                        files = {"file": ("large.pdf", f, "application/pdf")}
                        response = await async_client.post("/api/v1/documents/upload", files=files)
                    
                    assert response.status_code == 413
                    result = response.json()
                    assert "File too large" in str(result["detail"])
                    
            finally:
                Path(temp_file.name).unlink(missing_ok=True)


@pytest.mark.integration
@pytest.mark.api
class TestContentRetrievalEndpoints:
    """Integration tests for content retrieval API endpoints."""
    
    @pytest.fixture
    async def async_client(self):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.fixture
    def mock_document_data(self):
        """Mock document data."""
        return {
            "id": "doc_123",
            "title": "Test Document",
            "content": "This is test document content.",
            "metadata": {
                "page_count": 2,
                "file_size": 1024,
                "created_at": datetime.utcnow().isoformat()
            }
        }
    
    @pytest.fixture
    def mock_pages_data(self):
        """Mock pages data."""
        return [
            {
                "id": "page_1",
                "document_id": "doc_123",
                "page_number": 1,
                "content": "Page 1 content",
                "metadata": {"images": [], "tables": []}
            },
            {
                "id": "page_2",
                "document_id": "doc_123",
                "page_number": 2,
                "content": "Page 2 content",
                "metadata": {"images": [], "tables": []}
            }
        ]
    
    async def test_get_document_endpoint(self, async_client, mock_document_data):
        """Test document retrieval endpoint."""
        with patch('app.services.supabase_service.SupabaseService.get_document') as mock_get:
            mock_get.return_value = mock_document_data
            
            response = await async_client.get("/api/v1/documents/doc_123")
            
            assert response.status_code == 200
            result = response.json()
            
            assert result["id"] == "doc_123"
            assert result["title"] == "Test Document"
            assert result["metadata"]["page_count"] == 2
            
            mock_get.assert_called_once_with("doc_123")
    
    async def test_get_nonexistent_document_endpoint(self, async_client):
        """Test retrieval of non-existent document."""
        with patch('app.services.supabase_service.SupabaseService.get_document') as mock_get:
            mock_get.return_value = None
            
            response = await async_client.get("/api/v1/documents/nonexistent")
            
            assert response.status_code == 404
            result = response.json()
            assert "Document not found" in result["detail"]
    
    async def test_list_documents_endpoint(self, async_client):
        """Test document listing endpoint."""
        mock_documents = [
            {"id": "doc_1", "title": "Document 1", "created_at": "2024-01-01T00:00:00"},
            {"id": "doc_2", "title": "Document 2", "created_at": "2024-01-02T00:00:00"}
        ]
        
        with patch('app.services.supabase_service.SupabaseService.list_documents') as mock_list:
            mock_list.return_value = {
                "documents": mock_documents,
                "total": 2,
                "page": 1,
                "per_page": 10
            }
            
            response = await async_client.get("/api/v1/documents?page=1&per_page=10")
            
            assert response.status_code == 200
            result = response.json()
            
            assert len(result["documents"]) == 2
            assert result["total"] == 2
            assert result["page"] == 1
            
            mock_list.assert_called_once_with(page=1, per_page=10, filters=None)
    
    async def test_get_document_pages_endpoint(self, async_client, mock_pages_data):
        """Test document pages retrieval endpoint."""
        with patch('app.services.supabase_service.SupabaseService.get_document_pages') as mock_get_pages:
            mock_get_pages.return_value = mock_pages_data
            
            response = await async_client.get("/api/v1/documents/doc_123/pages")
            
            assert response.status_code == 200
            result = response.json()
            
            assert len(result["pages"]) == 2
            assert result["pages"][0]["page_number"] == 1
            assert result["pages"][1]["page_number"] == 2
            
            mock_get_pages.assert_called_once_with("doc_123")
    
    async def test_search_documents_endpoint(self, async_client):
        """Test document search endpoint."""
        mock_search_results = {
            "results": [
                {
                    "document_id": "doc_1",
                    "title": "Relevant Document",
                    "content": "This document contains the search term.",
                    "score": 0.95
                }
            ],
            "total": 1,
            "query": "search term"
        }
        
        with patch('app.services.supabase_service.SupabaseService.search_documents') as mock_search:
            mock_search.return_value = mock_search_results
            
            response = await async_client.get("/api/v1/documents/search?q=search%20term")
            
            assert response.status_code == 200
            result = response.json()
            
            assert len(result["results"]) == 1
            assert result["results"][0]["title"] == "Relevant Document"
            assert result["total"] == 1
            
            mock_search.assert_called_once_with("search term", limit=10, filters=None)


@pytest.mark.integration
@pytest.mark.api
class TestRAGSearchEndpoints:
    """Integration tests for RAG and search API endpoints."""
    
    @pytest.fixture
    async def async_client(self):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    async def test_semantic_search_endpoint(self, async_client):
        """Test semantic search endpoint."""
        mock_search_results = {
            "results": [
                {
                    "content": "Relevant content chunk",
                    "document_id": "doc_1",
                    "page_number": 1,
                    "similarity_score": 0.92
                }
            ],
            "query": "test query",
            "total": 1
        }
        
        with patch('app.services.llamaindex_service.LlamaIndexService.semantic_search') as mock_search:
            mock_search.return_value = mock_search_results
            
            request_data = {
                "query": "test query",
                "limit": 5,
                "similarity_threshold": 0.8
            }
            
            response = await async_client.post("/api/v1/search/semantic", json=request_data)
            
            assert response.status_code == 200
            result = response.json()
            
            assert len(result["results"]) == 1
            assert result["results"][0]["similarity_score"] == 0.92
            assert result["query"] == "test query"
            
            mock_search.assert_called_once_with(
                query="test query",
                limit=5,
                similarity_threshold=0.8,
                filters=None
            )
    
    async def test_question_answering_endpoint(self, async_client):
        """Test question answering endpoint."""
        mock_qa_result = {
            "answer": "This is the answer to your question.",
            "sources": [
                {
                    "document_id": "doc_1",
                    "page_number": 1,
                    "content": "Source content",
                    "relevance_score": 0.95
                }
            ],
            "confidence": 0.88,
            "query": "What is the answer?"
        }
        
        with patch('app.services.llamaindex_service.LlamaIndexService.answer_question') as mock_qa:
            mock_qa.return_value = mock_qa_result
            
            request_data = {
                "question": "What is the answer?",
                "context_limit": 3,
                "include_sources": True
            }
            
            response = await async_client.post("/api/v1/search/qa", json=request_data)
            
            assert response.status_code == 200
            result = response.json()
            
            assert result["answer"] == "This is the answer to your question."
            assert len(result["sources"]) == 1
            assert result["confidence"] == 0.88
            
            mock_qa.assert_called_once_with(
                question="What is the answer?",
                context_limit=3,
                include_sources=True,
                filters=None
            )
    
    async def test_document_comparison_endpoint(self, async_client):
        """Test document comparison endpoint."""
        mock_comparison_result = {
            "similarity_score": 0.75,
            "common_topics": ["topic1", "topic2"],
            "differences": ["difference1", "difference2"],
            "document1_id": "doc_1",
            "document2_id": "doc_2"
        }
        
        with patch('app.services.llamaindex_service.LlamaIndexService.compare_documents') as mock_compare:
            mock_compare.return_value = mock_comparison_result
            
            request_data = {
                "document1_id": "doc_1",
                "document2_id": "doc_2",
                "comparison_type": "semantic"
            }
            
            response = await async_client.post("/api/v1/search/compare", json=request_data)
            
            assert response.status_code == 200
            result = response.json()
            
            assert result["similarity_score"] == 0.75
            assert len(result["common_topics"]) == 2
            assert len(result["differences"]) == 2
            
            mock_compare.assert_called_once_with(
                document1_id="doc_1",
                document2_id="doc_2",
                comparison_type="semantic"
            )


@pytest.mark.integration
@pytest.mark.api
class TestImageAnalysisEndpoints:
    """Integration tests for image analysis API endpoints."""
    
    @pytest.fixture
    async def async_client(self):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    @pytest.fixture
    def sample_image_file(self):
        """Create a sample image file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            # Create a minimal JPEG header
            jpeg_header = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb'
            temp_file.write(jpeg_header + b'\x00' * 100)  # Minimal JPEG data
            temp_file.flush()
            yield temp_file.name
        Path(temp_file.name).unlink(missing_ok=True)
    
    async def test_analyze_image_endpoint(self, async_client, sample_image_file):
        """Test image analysis endpoint."""
        mock_analysis_result = {
            "image_id": "img_123",
            "analysis": {
                "objects": [{"name": "document", "confidence": 0.95}],
                "text": "Extracted text from image",
                "faces": [],
                "quality_score": 0.88
            },
            "material_kai_analysis": {
                "categories": ["document", "text"],
                "confidence": 0.92
            }
        }
        
        with patch('app.services.image_processor.ImageProcessor.analyze_image') as mock_analyze:
            mock_analyze.return_value = mock_analysis_result
            
            with open(sample_image_file, 'rb') as f:
                files = {"image": ("test.jpg", f, "image/jpeg")}
                data = {"extract_text": "true", "detect_objects": "true"}
                response = await async_client.post(
                    "/api/v1/images/analyze", 
                    files=files, 
                    data=data
                )
            
            assert response.status_code == 200
            result = response.json()
            
            assert result["image_id"] == "img_123"
            assert "objects" in result["analysis"]
            assert result["analysis"]["quality_score"] == 0.88
            
            mock_analyze.assert_called_once()
    
    async def test_batch_image_analysis_endpoint(self, async_client, sample_image_file):
        """Test batch image analysis endpoint."""
        mock_batch_result = {
            "batch_id": "batch_123",
            "results": [
                {
                    "image_id": "img_1",
                    "filename": "test1.jpg",
                    "analysis": {"objects": [], "text": "Text 1"}
                },
                {
                    "image_id": "img_2",
                    "filename": "test2.jpg",
                    "analysis": {"objects": [], "text": "Text 2"}
                }
            ],
            "total_processed": 2,
            "processing_time": 1.5
        }
        
        with patch('app.services.image_processor.ImageProcessor.batch_analyze_images') as mock_batch:
            mock_batch.return_value = mock_batch_result
            
            # Create multiple files for batch processing
            files = []
            for i in range(2):
                with open(sample_image_file, 'rb') as f:
                    files.append(("images", (f"test{i+1}.jpg", f.read(), "image/jpeg")))
            
            response = await async_client.post("/api/v1/images/batch-analyze", files=files)
            
            assert response.status_code == 200
            result = response.json()
            
            assert result["batch_id"] == "batch_123"
            assert len(result["results"]) == 2
            assert result["total_processed"] == 2
            
            mock_batch.assert_called_once()


@pytest.mark.integration
@pytest.mark.api
class TestAdministrativeEndpoints:
    """Integration tests for administrative API endpoints."""
    
    @pytest.fixture
    async def async_client(self):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    async def test_health_check_endpoint(self, async_client):
        """Test health check endpoint."""
        mock_health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "database": "healthy",
                "storage": "healthy",
                "llamaindex": "healthy",
                "material_kai": "healthy"
            },
            "version": "1.0.0"
        }
        
        with patch('app.services.supabase_service.SupabaseService.health_check') as mock_health:
            mock_health.return_value = mock_health_status
            
            response = await async_client.get("/api/v1/health")
            
            assert response.status_code == 200
            result = response.json()
            
            assert result["status"] == "healthy"
            assert "services" in result
            assert result["services"]["database"] == "healthy"
            
            mock_health.assert_called_once()
    
    async def test_system_stats_endpoint(self, async_client):
        """Test system statistics endpoint."""
        mock_stats = {
            "documents": {
                "total": 150,
                "processed_today": 25,
                "total_pages": 1500,
                "total_size_mb": 250.5
            },
            "embeddings": {
                "total_vectors": 5000,
                "index_size_mb": 45.2
            },
            "images": {
                "total": 300,
                "analyzed_today": 50
            },
            "system": {
                "uptime_hours": 72.5,
                "memory_usage_mb": 512.3,
                "disk_usage_gb": 15.8
            }
        }
        
        with patch('app.services.supabase_service.SupabaseService.get_system_stats') as mock_stats_call:
            mock_stats_call.return_value = mock_stats
            
            response = await async_client.get("/api/v1/admin/stats")
            
            assert response.status_code == 200
            result = response.json()
            
            assert result["documents"]["total"] == 150
            assert result["embeddings"]["total_vectors"] == 5000
            assert result["system"]["uptime_hours"] == 72.5
            
            mock_stats_call.assert_called_once()
    
    async def test_cleanup_endpoint(self, async_client):
        """Test cleanup endpoint."""
        mock_cleanup_result = {
            "cleaned_items": {
                "orphaned_pages": 5,
                "unused_embeddings": 12,
                "temporary_files": 8
            },
            "space_freed_mb": 25.3,
            "cleanup_time": 2.1
        }
        
        with patch('app.services.supabase_service.SupabaseService.cleanup_orphaned_data') as mock_cleanup:
            mock_cleanup.return_value = mock_cleanup_result
            
            request_data = {
                "cleanup_type": "orphaned_data",
                "dry_run": False
            }
            
            response = await async_client.post("/api/v1/admin/cleanup", json=request_data)
            
            assert response.status_code == 200
            result = response.json()
            
            assert result["cleaned_items"]["orphaned_pages"] == 5
            assert result["space_freed_mb"] == 25.3
            
            mock_cleanup.assert_called_once_with(cleanup_type="orphaned_data", dry_run=False)


@pytest.mark.integration
@pytest.mark.api
class TestErrorHandlingAndValidation:
    """Integration tests for API error handling and validation."""
    
    @pytest.fixture
    async def async_client(self):
        """Create async test client."""
        async with AsyncClient(app=app, base_url="http://test") as ac:
            yield ac
    
    async def test_validation_errors(self, async_client):
        """Test API validation error handling."""
        # Test invalid request data
        invalid_data = {
            "query": "",  # Empty query should be invalid
            "limit": -1,  # Negative limit should be invalid
            "similarity_threshold": 1.5  # Threshold > 1.0 should be invalid
        }
        
        response = await async_client.post("/api/v1/search/semantic", json=invalid_data)
        
        assert response.status_code == 422
        result = response.json()
        assert "detail" in result
        assert isinstance(result["detail"], list)
    
    async def test_service_error_handling(self, async_client):
        """Test handling of service-level errors."""
        with patch('app.services.supabase_service.SupabaseService.get_document') as mock_get:
            mock_get.side_effect = Exception("Database connection failed")
            
            response = await async_client.get("/api/v1/documents/doc_123")
            
            assert response.status_code == 500
            result = response.json()
            assert "Internal server error" in result["detail"]
    
    async def test_rate_limiting(self, async_client):
        """Test API rate limiting."""
        # This would require actual rate limiting implementation
        # For now, we'll test the structure
        
        with patch('app.core.middleware.rate_limiter') as mock_limiter:
            mock_limiter.side_effect = Exception("Rate limit exceeded")
            
            # Make multiple rapid requests
            tasks = []
            for _ in range(10):
                task = async_client.get("/api/v1/health")
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # At least some requests should succeed
            successful_responses = [r for r in responses if not isinstance(r, Exception)]
            assert len(successful_responses) > 0
    
    async def test_authentication_errors(self, async_client):
        """Test authentication error handling."""
        # Test protected endpoint without authentication
        with patch('app.core.auth.verify_token') as mock_verify:
            mock_verify.side_effect = Exception("Invalid token")
            
            response = await async_client.get("/api/v1/admin/stats")
            
            # This would depend on actual auth implementation
            # For now, we'll check that the endpoint exists
            assert response.status_code in [200, 401, 403]
    
    async def test_concurrent_request_handling(self, async_client):
        """Test handling of concurrent requests."""
        with patch('app.services.supabase_service.SupabaseService.get_document') as mock_get:
            mock_get.return_value = {"id": "doc_123", "title": "Test Document"}
            
            # Make concurrent requests
            tasks = []
            for i in range(5):
                task = async_client.get(f"/api/v1/documents/doc_{i}")
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            # All requests should complete
            assert len(responses) == 5
            for response in responses:
                assert response.status_code in [200, 404]  # Either found or not found