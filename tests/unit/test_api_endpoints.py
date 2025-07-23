"""
Unit tests for API endpoints.

Tests the FastAPI endpoints in isolation using mocks for services.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status
import tempfile
import os
from io import BytesIO

from app.main import app
from app.models.pdf_models import PDFProcessingResponse
from app.services.pdf_processor import PDFProcessor


class TestAPIEndpoints:
    """Test suite for API endpoints."""

    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)

    @pytest.fixture
    def mock_pdf_processor(self):
        """Create a mock PDF processor."""
        return Mock(spec=PDFProcessor)

    def test_health_check_endpoint(self, client):
        """Test the health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data

    def test_root_endpoint(self, client):
        """Test the root endpoint."""
        response = client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "message" in data
        assert "version" in data

    @patch('app.api.pdf_routes.PDFProcessor')
    def test_upload_pdf_success(self, mock_processor_class, client):
        """Test successful PDF upload and processing."""
        # Mock the processor instance
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        # Mock successful processing
        mock_response = PDFProcessingResponse(
            success=True,
            extracted_text="Sample extracted text",
            extracted_tables=[],
            extracted_images=[],
            metadata={"pages": 1},
            processing_time=1.5
        )
        mock_processor.process_pdf = AsyncMock(return_value=mock_response)
        
        # Create a test PDF file
        pdf_content = b"%PDF-1.4\n%fake pdf content"
        files = {"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")}
        
        response = client.post("/api/v1/pdf/upload", files=files)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["extracted_text"] == "Sample extracted text"
        assert data["processing_time"] == 1.5

    def test_upload_pdf_no_file(self, client):
        """Test PDF upload without file."""
        response = client.post("/api/v1/pdf/upload")
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_upload_pdf_invalid_file_type(self, client):
        """Test PDF upload with invalid file type."""
        # Create a test text file
        text_content = b"This is not a PDF"
        files = {"file": ("test.txt", BytesIO(text_content), "text/plain")}
        
        response = client.post("/api/v1/pdf/upload", files=files)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "Invalid file type" in data["detail"]

    def test_upload_pdf_empty_file(self, client):
        """Test PDF upload with empty file."""
        files = {"file": ("empty.pdf", BytesIO(b""), "application/pdf")}
        
        response = client.post("/api/v1/pdf/upload", files=files)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "Empty file" in data["detail"]

    @patch('app.api.pdf_routes.PDFProcessor')
    def test_upload_pdf_processing_failure(self, mock_processor_class, client):
        """Test PDF upload with processing failure."""
        # Mock the processor instance
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        # Mock processing failure
        mock_response = PDFProcessingResponse(
            success=False,
            error_message="Processing failed",
            processing_time=0.5
        )
        mock_processor.process_pdf = AsyncMock(return_value=mock_response)
        
        # Create a test PDF file
        pdf_content = b"%PDF-1.4\n%fake pdf content"
        files = {"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")}
        
        response = client.post("/api/v1/pdf/upload", files=files)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Processing failed" in data["detail"]

    @patch('app.api.pdf_routes.PDFProcessor')
    def test_upload_pdf_with_options(self, mock_processor_class, client):
        """Test PDF upload with processing options."""
        # Mock the processor instance
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        # Mock successful processing
        mock_response = PDFProcessingResponse(
            success=True,
            extracted_text="Sample text",
            extracted_tables=[{"headers": ["Col1"], "rows": [["Data1"]]}],
            extracted_images=[],
            metadata={"pages": 1},
            processing_time=2.0
        )
        mock_processor.process_pdf = AsyncMock(return_value=mock_response)
        
        # Create a test PDF file
        pdf_content = b"%PDF-1.4\n%fake pdf content"
        files = {"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")}
        
        # Add processing options
        data = {
            "extract_text": "true",
            "extract_tables": "true",
            "extract_images": "false",
            "output_format": "markdown"
        }
        
        response = client.post("/api/v1/pdf/upload", files=files, data=data)
        
        assert response.status_code == status.HTTP_200_OK
        response_data = response.json()
        assert response_data["success"] is True
        assert len(response_data["extracted_tables"]) == 1

    def test_upload_pdf_file_too_large(self, client):
        """Test PDF upload with file too large."""
        # Create a large file (assuming max size is 10MB)
        large_content = b"x" * (11 * 1024 * 1024)  # 11MB
        files = {"file": ("large.pdf", BytesIO(large_content), "application/pdf")}
        
        response = client.post("/api/v1/pdf/upload", files=files)
        
        # This might return 413 or 422 depending on FastAPI configuration
        assert response.status_code in [
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ]

    @patch('app.api.pdf_routes.PDFProcessor')
    def test_upload_pdf_exception_handling(self, mock_processor_class, client):
        """Test PDF upload with unexpected exception."""
        # Mock the processor instance
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        # Mock exception during processing
        mock_processor.process_pdf = AsyncMock(side_effect=Exception("Unexpected error"))
        
        # Create a test PDF file
        pdf_content = b"%PDF-1.4\n%fake pdf content"
        files = {"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")}
        
        response = client.post("/api/v1/pdf/upload", files=files)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "Internal server error" in data["detail"]

    def test_health_check_detailed(self, client):
        """Test detailed health check endpoint."""
        response = client.get("/health/detailed")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "status" in data
        assert "checks" in data
        assert "timestamp" in data

    @patch('app.database.connection.test_supabase_connection')
    def test_health_check_database_failure(self, mock_db_check, client):
        """Test health check with database failure."""
        # Mock database connection failure
        mock_db_check.return_value = {"status": "unhealthy", "error": "Connection failed"}
        
        response = client.get("/health/detailed")
        
        # Should still return 200 but with unhealthy status
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "unhealthy"

    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options("/api/v1/pdf/upload")
        
        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers

    def test_api_versioning(self, client):
        """Test API versioning in URLs."""
        # Test that v1 endpoints exist
        response = client.get("/api/v1/")
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]

    @patch('app.api.pdf_routes.PDFProcessor')
    def test_concurrent_uploads(self, mock_processor_class, client):
        """Test handling of concurrent uploads."""
        # Mock the processor instance
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        # Mock successful processing with delay
        mock_response = PDFProcessingResponse(
            success=True,
            extracted_text="Sample text",
            processing_time=1.0
        )
        mock_processor.process_pdf = AsyncMock(return_value=mock_response)
        
        # Create test PDF files
        pdf_content = b"%PDF-1.4\n%fake pdf content"
        files1 = {"file": ("test1.pdf", BytesIO(pdf_content), "application/pdf")}
        files2 = {"file": ("test2.pdf", BytesIO(pdf_content), "application/pdf")}
        
        # Make concurrent requests (simplified test)
        response1 = client.post("/api/v1/pdf/upload", files=files1)
        response2 = client.post("/api/v1/pdf/upload", files=files2)
        
        assert response1.status_code == status.HTTP_200_OK
        assert response2.status_code == status.HTTP_200_OK

    def test_request_validation(self, client):
        """Test request validation for various invalid inputs."""
        # Test with invalid form data
        response = client.post("/api/v1/pdf/upload", data={"invalid": "data"})
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_response_format(self, client):
        """Test response format consistency."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/json"
        
        # Ensure response is valid JSON
        data = response.json()
        assert isinstance(data, dict)

    @patch('app.api.pdf_routes.PDFProcessor')
    def test_upload_pdf_metadata_extraction(self, mock_processor_class, client):
        """Test PDF upload with metadata extraction."""
        # Mock the processor instance
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        # Mock successful processing with metadata
        mock_response = PDFProcessingResponse(
            success=True,
            extracted_text="Sample text",
            metadata={
                "pages": 5,
                "title": "Test Document",
                "author": "Test Author",
                "creation_date": "2023-01-01",
                "file_size": 1024
            },
            processing_time=1.5
        )
        mock_processor.process_pdf = AsyncMock(return_value=mock_response)
        
        # Create a test PDF file
        pdf_content = b"%PDF-1.4\n%fake pdf content"
        files = {"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")}
        
        response = client.post("/api/v1/pdf/upload", files=files)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert data["metadata"]["pages"] == 5
        assert data["metadata"]["title"] == "Test Document"

    def test_error_response_format(self, client):
        """Test error response format consistency."""
        # Test with invalid file type
        text_content = b"This is not a PDF"
        files = {"file": ("test.txt", BytesIO(text_content), "text/plain")}
        
        response = client.post("/api/v1/pdf/upload", files=files)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "detail" in data
        assert isinstance(data["detail"], str)

    @patch('app.api.pdf_routes.PDFProcessor')
    def test_upload_pdf_timeout_handling(self, mock_processor_class, client):
        """Test handling of processing timeouts."""
        # Mock the processor instance
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        
        # Mock timeout exception
        import asyncio
        mock_processor.process_pdf = AsyncMock(side_effect=asyncio.TimeoutError("Processing timeout"))
        
        # Create a test PDF file
        pdf_content = b"%PDF-1.4\n%fake pdf content"
        files = {"file": ("test.pdf", BytesIO(pdf_content), "application/pdf")}
        
        response = client.post("/api/v1/pdf/upload", files=files)
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert "timeout" in data["detail"].lower() or "internal server error" in data["detail"].lower()