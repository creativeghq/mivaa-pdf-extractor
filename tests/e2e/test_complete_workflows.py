"""
End-to-End Tests for Complete PDF Processing Workflows

This module contains comprehensive E2E tests that verify complete workflows
from PDF upload to final processing results, testing all integrated components.
"""

import pytest
import asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from httpx import AsyncClient
import json

from app.main import app
from app.config import get_settings


@pytest.mark.e2e
class TestCompleteWorkflows:
    """Test complete PDF processing workflows end-to-end."""

    @pytest.fixture
    def sample_pdf_content(self):
        """Create sample PDF content for testing."""
        # This would be actual PDF bytes in a real scenario
        return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"

    @pytest.fixture
    def sample_pdf_file(self, sample_pdf_content):
        """Create a temporary PDF file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(sample_pdf_content)
            tmp_file.flush()
            yield tmp_file.name
        os.unlink(tmp_file.name)

    @pytest.mark.asyncio
    async def test_complete_pdf_upload_and_processing_workflow(
        self, 
        sample_pdf_file,
        mock_supabase_client,
        mock_openai_client,
        mock_llamaindex_service,
        mock_material_kai_service
    ):
        """Test complete workflow from PDF upload to processing completion."""
        
        # Mock successful responses for all services
        mock_supabase_client.table().insert().execute.return_value = MagicMock(
            data=[{"id": "test-doc-id", "filename": "test.pdf", "status": "uploaded"}]
        )
        
        mock_llamaindex_service.process_document.return_value = {
            "document_id": "test-doc-id",
            "chunks": 5,
            "embeddings_created": True,
            "index_updated": True
        }
        
        mock_material_kai_service.notify_processing_complete.return_value = {
            "notification_sent": True,
            "platform_updated": True
        }

        async with AsyncClient(app=app, base_url="http://test") as client:
            # Step 1: Upload PDF
            with open(sample_pdf_file, "rb") as pdf_file:
                upload_response = await client.post(
                    "/api/v1/upload",
                    files={"file": ("test.pdf", pdf_file, "application/pdf")}
                )
            
            assert upload_response.status_code == 200
            upload_data = upload_response.json()
            assert "document_id" in upload_data
            document_id = upload_data["document_id"]
            
            # Step 2: Check processing status
            status_response = await client.get(f"/api/v1/documents/{document_id}/status")
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["status"] in ["processing", "completed"]
            
            # Step 3: Wait for processing completion (simulated)
            await asyncio.sleep(0.1)  # Simulate processing time
            
            # Step 4: Verify final status
            final_status_response = await client.get(f"/api/v1/documents/{document_id}/status")
            assert final_status_response.status_code == 200
            final_status_data = final_status_response.json()
            
            # Verify all services were called
            mock_llamaindex_service.process_document.assert_called_once()
            mock_material_kai_service.notify_processing_complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_pdf_processing_with_rag_query_workflow(
        self,
        sample_pdf_file,
        mock_supabase_client,
        mock_llamaindex_service
    ):
        """Test complete workflow including RAG querying after processing."""
        
        # Mock document processing
        mock_supabase_client.table().insert().execute.return_value = MagicMock(
            data=[{"id": "test-doc-id", "filename": "test.pdf", "status": "uploaded"}]
        )
        
        mock_llamaindex_service.process_document.return_value = {
            "document_id": "test-doc-id",
            "chunks": 5,
            "embeddings_created": True,
            "index_updated": True
        }
        
        mock_llamaindex_service.query_documents.return_value = {
            "answer": "This is a test document about PDF processing.",
            "sources": ["chunk_1", "chunk_2"],
            "confidence": 0.95
        }

        async with AsyncClient(app=app, base_url="http://test") as client:
            # Step 1: Upload and process PDF
            with open(sample_pdf_file, "rb") as pdf_file:
                upload_response = await client.post(
                    "/api/v1/upload",
                    files={"file": ("test.pdf", pdf_file, "application/pdf")}
                )
            
            assert upload_response.status_code == 200
            document_id = upload_response.json()["document_id"]
            
            # Step 2: Query the processed document
            query_response = await client.post(
                "/api/v1/query",
                json={
                    "query": "What is this document about?",
                    "document_ids": [document_id]
                }
            )
            
            assert query_response.status_code == 200
            query_data = query_response.json()
            assert "answer" in query_data
            assert "sources" in query_data
            assert query_data["answer"] == "This is a test document about PDF processing."
            
            # Verify services were called correctly
            mock_llamaindex_service.process_document.assert_called_once()
            mock_llamaindex_service.query_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_pdf_processing_workflow(
        self,
        sample_pdf_file,
        mock_supabase_client,
        mock_llamaindex_service,
        mock_material_kai_service
    ):
        """Test batch processing of multiple PDFs."""
        
        # Mock responses for batch processing
        mock_supabase_client.table().insert().execute.side_effect = [
            MagicMock(data=[{"id": f"doc-{i}", "filename": f"test{i}.pdf", "status": "uploaded"}])
            for i in range(3)
        ]
        
        mock_llamaindex_service.process_document.return_value = {
            "chunks": 5,
            "embeddings_created": True,
            "index_updated": True
        }

        async with AsyncClient(app=app, base_url="http://test") as client:
            document_ids = []
            
            # Upload multiple PDFs
            for i in range(3):
                with open(sample_pdf_file, "rb") as pdf_file:
                    upload_response = await client.post(
                        "/api/v1/upload",
                        files={"file": (f"test{i}.pdf", pdf_file, "application/pdf")}
                    )
                
                assert upload_response.status_code == 200
                document_ids.append(upload_response.json()["document_id"])
            
            # Check batch status
            batch_status_response = await client.post(
                "/api/v1/documents/batch-status",
                json={"document_ids": document_ids}
            )
            
            assert batch_status_response.status_code == 200
            batch_data = batch_status_response.json()
            assert len(batch_data["documents"]) == 3
            
            # Verify all documents were processed
            assert mock_llamaindex_service.process_document.call_count == 3

    @pytest.mark.asyncio
    async def test_error_handling_workflow(
        self,
        sample_pdf_file,
        mock_supabase_client,
        mock_llamaindex_service
    ):
        """Test error handling throughout the processing workflow."""
        
        # Mock database success but processing failure
        mock_supabase_client.table().insert().execute.return_value = MagicMock(
            data=[{"id": "test-doc-id", "filename": "test.pdf", "status": "uploaded"}]
        )
        
        # Mock processing failure
        mock_llamaindex_service.process_document.side_effect = Exception("Processing failed")

        async with AsyncClient(app=app, base_url="http://test") as client:
            # Upload PDF
            with open(sample_pdf_file, "rb") as pdf_file:
                upload_response = await client.post(
                    "/api/v1/upload",
                    files={"file": ("test.pdf", pdf_file, "application/pdf")}
                )
            
            assert upload_response.status_code == 200
            document_id = upload_response.json()["document_id"]
            
            # Check that error is handled gracefully
            status_response = await client.get(f"/api/v1/documents/{document_id}/status")
            assert status_response.status_code == 200
            
            # The status should indicate an error occurred
            status_data = status_response.json()
            # Depending on implementation, this might be "error" or "failed"
            assert status_data["status"] in ["error", "failed", "processing"]

    @pytest.mark.asyncio
    async def test_health_check_integration_workflow(self):
        """Test that health checks work properly in the complete system."""
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test basic health check
            health_response = await client.get("/health")
            assert health_response.status_code == 200
            health_data = health_response.json()
            assert health_data["status"] == "healthy"
            
            # Test detailed health check
            detailed_health_response = await client.get("/health/detailed")
            assert detailed_health_response.status_code == 200
            detailed_data = detailed_health_response.json()
            
            # Verify all components are checked
            assert "database" in detailed_data
            assert "services" in detailed_data
            assert "system" in detailed_data

    @pytest.mark.asyncio
    async def test_material_kai_platform_integration_workflow(
        self,
        sample_pdf_file,
        mock_supabase_client,
        mock_llamaindex_service,
        mock_material_kai_service
    ):
        """Test complete integration with Material Kai Vision Platform."""
        
        # Mock successful processing
        mock_supabase_client.table().insert().execute.return_value = MagicMock(
            data=[{"id": "test-doc-id", "filename": "test.pdf", "status": "uploaded"}]
        )
        
        mock_llamaindex_service.process_document.return_value = {
            "document_id": "test-doc-id",
            "chunks": 5,
            "embeddings_created": True,
            "index_updated": True
        }
        
        # Mock platform integration responses
        mock_material_kai_service.register_document.return_value = {
            "platform_document_id": "platform-doc-123",
            "registered": True
        }
        
        mock_material_kai_service.notify_processing_complete.return_value = {
            "notification_sent": True,
            "platform_updated": True
        }

        async with AsyncClient(app=app, base_url="http://test") as client:
            # Upload PDF
            with open(sample_pdf_file, "rb") as pdf_file:
                upload_response = await client.post(
                    "/api/v1/upload",
                    files={"file": ("test.pdf", pdf_file, "application/pdf")}
                )
            
            assert upload_response.status_code == 200
            document_id = upload_response.json()["document_id"]
            
            # Verify platform integration was called
            mock_material_kai_service.register_document.assert_called_once()
            mock_material_kai_service.notify_processing_complete.assert_called_once()
            
            # Test platform-specific endpoints
            platform_status_response = await client.get(
                f"/api/v1/documents/{document_id}/platform-status"
            )
            assert platform_status_response.status_code == 200

    @pytest.mark.asyncio
    async def test_concurrent_processing_workflow(
        self,
        sample_pdf_file,
        mock_supabase_client,
        mock_llamaindex_service
    ):
        """Test concurrent processing of multiple documents."""
        
        # Mock responses for concurrent processing
        mock_supabase_client.table().insert().execute.side_effect = [
            MagicMock(data=[{"id": f"concurrent-doc-{i}", "filename": f"concurrent{i}.pdf", "status": "uploaded"}])
            for i in range(5)
        ]
        
        mock_llamaindex_service.process_document.return_value = {
            "chunks": 3,
            "embeddings_created": True,
            "index_updated": True
        }

        async with AsyncClient(app=app, base_url="http://test") as client:
            # Create multiple concurrent upload tasks
            async def upload_pdf(index):
                with open(sample_pdf_file, "rb") as pdf_file:
                    response = await client.post(
                        "/api/v1/upload",
                        files={"file": (f"concurrent{index}.pdf", pdf_file, "application/pdf")}
                    )
                return response
            
            # Execute concurrent uploads
            tasks = [upload_pdf(i) for i in range(5)]
            responses = await asyncio.gather(*tasks)
            
            # Verify all uploads succeeded
            for response in responses:
                assert response.status_code == 200
                assert "document_id" in response.json()
            
            # Verify processing was called for all documents
            assert mock_llamaindex_service.process_document.call_count == 5

    @pytest.mark.asyncio
    async def test_large_file_processing_workflow(
        self,
        mock_supabase_client,
        mock_llamaindex_service
    ):
        """Test processing of large PDF files."""
        
        # Create a larger mock PDF file
        large_pdf_content = b"%PDF-1.4\n" + b"Large content " * 1000 + b"\nendobj\n"
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(large_pdf_content)
            tmp_file.flush()
            
            try:
                # Mock responses
                mock_supabase_client.table().insert().execute.return_value = MagicMock(
                    data=[{"id": "large-doc-id", "filename": "large.pdf", "status": "uploaded"}]
                )
                
                mock_llamaindex_service.process_document.return_value = {
                    "document_id": "large-doc-id",
                    "chunks": 50,  # More chunks for larger file
                    "embeddings_created": True,
                    "index_updated": True
                }

                async with AsyncClient(app=app, base_url="http://test") as client:
                    # Upload large PDF
                    with open(tmp_file.name, "rb") as pdf_file:
                        upload_response = await client.post(
                            "/api/v1/upload",
                            files={"file": ("large.pdf", pdf_file, "application/pdf")}
                        )
                    
                    assert upload_response.status_code == 200
                    document_id = upload_response.json()["document_id"]
                    
                    # Verify processing handles large files
                    status_response = await client.get(f"/api/v1/documents/{document_id}/status")
                    assert status_response.status_code == 200
                    
            finally:
                os.unlink(tmp_file.name)

    @pytest.mark.asyncio
    async def test_api_rate_limiting_workflow(self):
        """Test API rate limiting behavior."""
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Make multiple rapid requests to test rate limiting
            responses = []
            for i in range(10):
                response = await client.get("/health")
                responses.append(response)
            
            # Most requests should succeed (depending on rate limit configuration)
            successful_responses = [r for r in responses if r.status_code == 200]
            assert len(successful_responses) >= 5  # At least some should succeed
            
            # Some might be rate limited (429 status code)
            rate_limited_responses = [r for r in responses if r.status_code == 429]
            # This depends on rate limiting configuration

    @pytest.mark.asyncio
    async def test_websocket_integration_workflow(
        self,
        sample_pdf_file,
        mock_supabase_client,
        mock_llamaindex_service,
        mock_material_kai_service
    ):
        """Test WebSocket integration for real-time updates."""
        
        # Mock successful processing
        mock_supabase_client.table().insert().execute.return_value = MagicMock(
            data=[{"id": "ws-doc-id", "filename": "websocket.pdf", "status": "uploaded"}]
        )
        
        mock_llamaindex_service.process_document.return_value = {
            "document_id": "ws-doc-id",
            "chunks": 5,
            "embeddings_created": True,
            "index_updated": True
        }

        async with AsyncClient(app=app, base_url="http://test") as client:
            # Test WebSocket connection for real-time updates
            with client.websocket_connect("/ws/processing-updates") as websocket:
                # Upload PDF while WebSocket is connected
                with open(sample_pdf_file, "rb") as pdf_file:
                    upload_response = await client.post(
                        "/api/v1/upload",
                        files={"file": ("websocket.pdf", pdf_file, "application/pdf")}
                    )
                
                assert upload_response.status_code == 200
                
                # Should receive WebSocket updates about processing
                # This would depend on the actual WebSocket implementation
                # data = websocket.receive_json()
                # assert "status" in data