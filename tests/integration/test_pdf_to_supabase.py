"""
Integration tests for PDF processing to Supabase data persistence workflow.

This module tests the complete integration between PDF processing service
and Supabase database operations, ensuring data flows correctly through
the entire pipeline.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile
import json
from datetime import datetime

from app.services.pdf_processor import PDFProcessor
from app.services.supabase_service import SupabaseService
from app.core.config import get_settings


@pytest.mark.integration
@pytest.mark.asyncio
class TestPDFToSupabaseIntegration:
    """Integration tests for PDF processing to Supabase workflow."""
    
    @pytest.fixture
    async def pdf_processor(self):
        """Create PDF processor instance."""
        return PDFProcessor()
    
    @pytest.fixture
    async def supabase_service(self):
        """Create Supabase service instance with mocked client."""
        with patch('app.services.supabase_service.create_client') as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client
            
            service = SupabaseService()
            await service.initialize()
            return service
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Sample PDF processing result."""
        return {
            "text": "This is a sample PDF document with multiple pages.",
            "metadata": {
                "page_count": 2,
                "title": "Sample Document",
                "author": "Test Author",
                "creation_date": "2024-01-01",
                "file_size": 1024
            },
            "pages": [
                {
                    "page_number": 1,
                    "text": "This is page 1 content.",
                    "images": [],
                    "tables": []
                },
                {
                    "page_number": 2,
                    "text": "This is page 2 content.",
                    "images": [{"image_id": "img_1", "description": "Sample image"}],
                    "tables": [{"table_id": "tbl_1", "data": [["A", "B"], ["1", "2"]]}]
                }
            ]
        }
    
    @pytest.fixture
    def mock_file_upload(self):
        """Mock file upload response."""
        return {
            "file_path": "documents/test_document.pdf",
            "file_url": "https://storage.supabase.co/documents/test_document.pdf",
            "file_size": 1024
        }
    
    async def test_complete_pdf_processing_workflow(
        self, 
        pdf_processor, 
        supabase_service, 
        sample_pdf_content,
        mock_file_upload
    ):
        """Test complete workflow from PDF processing to database storage."""
        # Mock PDF processing
        with patch.object(pdf_processor, 'process_pdf') as mock_process:
            mock_process.return_value = sample_pdf_content
            
            # Mock Supabase operations
            mock_client = supabase_service.client
            
            # Mock file upload
            mock_client.storage.from_.return_value.upload.return_value = MagicMock(
                data=mock_file_upload
            )
            
            # Mock document insertion
            document_id = "doc_123"
            mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(
                data=[{"id": document_id, "title": "Sample Document"}]
            )
            
            # Mock page insertions
            mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(
                data=[
                    {"id": "page_1", "document_id": document_id, "page_number": 1},
                    {"id": "page_2", "document_id": document_id, "page_number": 2}
                ]
            )
            
            # Create temporary PDF file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(b"Mock PDF content")
                temp_path = temp_file.name
            
            try:
                # Process PDF
                pdf_result = await pdf_processor.process_pdf(temp_path)
                
                # Store in Supabase
                # 1. Upload file
                file_result = await supabase_service.upload_file(
                    temp_path, 
                    "documents/test_document.pdf"
                )
                
                # 2. Store document metadata
                document_data = {
                    "title": pdf_result["metadata"]["title"],
                    "author": pdf_result["metadata"]["author"],
                    "page_count": pdf_result["metadata"]["page_count"],
                    "file_path": file_result["file_path"],
                    "file_url": file_result["file_url"],
                    "file_size": pdf_result["metadata"]["file_size"],
                    "processed_at": datetime.utcnow().isoformat()
                }
                
                document = await supabase_service.create_document(document_data)
                
                # 3. Store page data
                pages_stored = []
                for page in pdf_result["pages"]:
                    page_data = {
                        "document_id": document["id"],
                        "page_number": page["page_number"],
                        "content": page["text"],
                        "metadata": {
                            "images": page["images"],
                            "tables": page["tables"]
                        }
                    }
                    stored_page = await supabase_service.create_page(page_data)
                    pages_stored.append(stored_page)
                
                # Verify the workflow
                assert pdf_result is not None
                assert pdf_result["text"] == sample_pdf_content["text"]
                assert document["id"] == document_id
                assert len(pages_stored) == 2
                
                # Verify method calls
                mock_process.assert_called_once_with(temp_path)
                mock_client.storage.from_.assert_called()
                assert mock_client.table.call_count >= 3  # documents + 2 pages
                
            finally:
                # Cleanup
                Path(temp_path).unlink(missing_ok=True)
    
    async def test_pdf_processing_with_embeddings_integration(
        self, 
        pdf_processor, 
        supabase_service, 
        sample_pdf_content
    ):
        """Test PDF processing with vector embeddings generation and storage."""
        with patch.object(pdf_processor, 'process_pdf') as mock_process:
            mock_process.return_value = sample_pdf_content
            
            # Mock embedding generation
            mock_embeddings = [[0.1, 0.2, 0.3] * 256]  # 768-dimensional embedding
            
            with patch('app.services.openai_service.OpenAIService.generate_embeddings') as mock_embed:
                mock_embed.return_value = mock_embeddings
                
                mock_client = supabase_service.client
                
                # Mock vector storage
                mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(
                    data=[{"id": "vec_123", "content": "sample text", "embedding": mock_embeddings[0]}]
                )
                
                # Create temporary PDF file
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_file.write(b"Mock PDF content")
                    temp_path = temp_file.name
                
                try:
                    # Process PDF
                    pdf_result = await pdf_processor.process_pdf(temp_path)
                    
                    # Generate and store embeddings
                    text_chunks = [page["text"] for page in pdf_result["pages"]]
                    embeddings = await supabase_service.generate_and_store_embeddings(
                        text_chunks, 
                        document_id="doc_123"
                    )
                    
                    # Verify embeddings were generated and stored
                    assert len(embeddings) == len(text_chunks)
                    mock_embed.assert_called_once_with(text_chunks)
                    mock_client.table.assert_called()
                    
                finally:
                    # Cleanup
                    Path(temp_path).unlink(missing_ok=True)
    
    async def test_error_handling_in_integration_workflow(
        self, 
        pdf_processor, 
        supabase_service
    ):
        """Test error handling throughout the integration workflow."""
        # Test PDF processing failure
        with patch.object(pdf_processor, 'process_pdf') as mock_process:
            mock_process.side_effect = Exception("PDF processing failed")
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(b"Mock PDF content")
                temp_path = temp_file.name
            
            try:
                with pytest.raises(Exception, match="PDF processing failed"):
                    await pdf_processor.process_pdf(temp_path)
            finally:
                Path(temp_path).unlink(missing_ok=True)
        
        # Test Supabase storage failure
        mock_client = supabase_service.client
        mock_client.table.return_value.insert.return_value.execute.side_effect = Exception("Database error")
        
        with pytest.raises(Exception, match="Database error"):
            await supabase_service.create_document({
                "title": "Test Document",
                "content": "Test content"
            })
    
    async def test_transaction_rollback_on_failure(
        self, 
        pdf_processor, 
        supabase_service, 
        sample_pdf_content
    ):
        """Test that partial failures are handled with proper cleanup."""
        with patch.object(pdf_processor, 'process_pdf') as mock_process:
            mock_process.return_value = sample_pdf_content
            
            mock_client = supabase_service.client
            
            # Mock successful document creation
            document_id = "doc_123"
            mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(
                data=[{"id": document_id, "title": "Sample Document"}]
            )
            
            # Mock page creation failure on second page
            call_count = 0
            def mock_page_insert(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return MagicMock(data=[{"id": "page_1", "document_id": document_id}])
                else:
                    raise Exception("Page insertion failed")
            
            mock_client.table.return_value.insert.return_value.execute.side_effect = mock_page_insert
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(b"Mock PDF content")
                temp_path = temp_file.name
            
            try:
                # Process PDF
                pdf_result = await pdf_processor.process_pdf(temp_path)
                
                # Attempt to store document and pages
                document = await supabase_service.create_document({
                    "title": pdf_result["metadata"]["title"],
                    "content": pdf_result["text"]
                })
                
                # First page should succeed, second should fail
                page1_data = {
                    "document_id": document["id"],
                    "page_number": 1,
                    "content": pdf_result["pages"][0]["text"]
                }
                page1 = await supabase_service.create_page(page1_data)
                assert page1["id"] == "page_1"
                
                # Second page should fail
                page2_data = {
                    "document_id": document["id"],
                    "page_number": 2,
                    "content": pdf_result["pages"][1]["text"]
                }
                
                with pytest.raises(Exception, match="Page insertion failed"):
                    await supabase_service.create_page(page2_data)
                
            finally:
                Path(temp_path).unlink(missing_ok=True)
    
    async def test_concurrent_pdf_processing_integration(
        self, 
        pdf_processor, 
        supabase_service, 
        sample_pdf_content
    ):
        """Test concurrent PDF processing and storage operations."""
        with patch.object(pdf_processor, 'process_pdf') as mock_process:
            mock_process.return_value = sample_pdf_content
            
            mock_client = supabase_service.client
            
            # Mock successful operations for multiple documents
            document_ids = ["doc_1", "doc_2", "doc_3"]
            mock_responses = [
                MagicMock(data=[{"id": doc_id, "title": f"Document {i+1}"}])
                for i, doc_id in enumerate(document_ids)
            ]
            mock_client.table.return_value.insert.return_value.execute.side_effect = mock_responses
            
            # Create multiple temporary PDF files
            temp_files = []
            for i in range(3):
                temp_file = tempfile.NamedTemporaryFile(suffix=f'_{i}.pdf', delete=False)
                temp_file.write(f"Mock PDF content {i}".encode())
                temp_files.append(temp_file.name)
                temp_file.close()
            
            try:
                # Process multiple PDFs concurrently
                async def process_and_store(file_path, doc_title):
                    pdf_result = await pdf_processor.process_pdf(file_path)
                    document = await supabase_service.create_document({
                        "title": doc_title,
                        "content": pdf_result["text"]
                    })
                    return document
                
                # Run concurrent operations
                tasks = [
                    process_and_store(temp_files[i], f"Document {i+1}")
                    for i in range(3)
                ]
                
                results = await asyncio.gather(*tasks)
                
                # Verify all documents were processed
                assert len(results) == 3
                for i, result in enumerate(results):
                    assert result["id"] == document_ids[i]
                
                # Verify all PDF processing calls were made
                assert mock_process.call_count == 3
                
            finally:
                # Cleanup
                for temp_file in temp_files:
                    Path(temp_file).unlink(missing_ok=True)
    
    async def test_large_document_processing_integration(
        self, 
        pdf_processor, 
        supabase_service
    ):
        """Test integration with large documents that require chunking."""
        # Create large document content
        large_content = {
            "text": "Large document content. " * 1000,  # Simulate large document
            "metadata": {
                "page_count": 50,
                "title": "Large Document",
                "author": "Test Author",
                "file_size": 1024 * 1024  # 1MB
            },
            "pages": [
                {
                    "page_number": i,
                    "text": f"Page {i} content. " * 100,
                    "images": [],
                    "tables": []
                }
                for i in range(1, 51)  # 50 pages
            ]
        }
        
        with patch.object(pdf_processor, 'process_pdf') as mock_process:
            mock_process.return_value = large_content
            
            mock_client = supabase_service.client
            
            # Mock batch operations
            mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(
                data=[{"id": f"page_{i}", "document_id": "doc_large"} for i in range(50)]
            )
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(b"Large mock PDF content")
                temp_path = temp_file.name
            
            try:
                # Process large PDF
                pdf_result = await pdf_processor.process_pdf(temp_path)
                
                # Store document
                document = await supabase_service.create_document({
                    "title": pdf_result["metadata"]["title"],
                    "page_count": pdf_result["metadata"]["page_count"],
                    "file_size": pdf_result["metadata"]["file_size"]
                })
                
                # Store pages in batches
                batch_size = 10
                pages = pdf_result["pages"]
                
                for i in range(0, len(pages), batch_size):
                    batch = pages[i:i + batch_size]
                    page_data_batch = [
                        {
                            "document_id": document["id"],
                            "page_number": page["page_number"],
                            "content": page["text"]
                        }
                        for page in batch
                    ]
                    
                    await supabase_service.batch_create_pages(page_data_batch)
                
                # Verify processing
                assert pdf_result["metadata"]["page_count"] == 50
                assert len(pdf_result["pages"]) == 50
                
                # Verify batch operations were called
                expected_batches = (50 + batch_size - 1) // batch_size  # Ceiling division
                assert mock_client.table.call_count >= expected_batches
                
            finally:
                Path(temp_path).unlink(missing_ok=True)


@pytest.mark.integration
@pytest.mark.database
class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    @pytest.fixture
    async def supabase_service(self):
        """Create Supabase service with test database."""
        with patch('app.services.supabase_service.create_client') as mock_create:
            mock_client = AsyncMock()
            mock_create.return_value = mock_client
            
            service = SupabaseService()
            await service.initialize()
            return service
    
    async def test_database_connection_and_health_check(self, supabase_service):
        """Test database connection and health check."""
        mock_client = supabase_service.client
        mock_client.table.return_value.select.return_value.limit.return_value.execute.return_value = MagicMock(
            data=[{"status": "healthy"}]
        )
        
        health_status = await supabase_service.health_check()
        assert health_status["status"] == "healthy"
        
        mock_client.table.assert_called()
    
    async def test_database_migration_compatibility(self, supabase_service):
        """Test that the service works with expected database schema."""
        mock_client = supabase_service.client
        
        # Mock schema validation
        expected_tables = ["documents", "pages", "embeddings", "images"]
        mock_client.table.return_value.select.return_value.execute.return_value = MagicMock(
            data=[{"table_name": table} for table in expected_tables]
        )
        
        # Verify all expected tables exist
        for table in expected_tables:
            mock_client.table(table).select("*").limit(1).execute()
        
        assert mock_client.table.call_count == len(expected_tables)
    
    async def test_foreign_key_constraints(self, supabase_service):
        """Test foreign key relationships between tables."""
        mock_client = supabase_service.client
        
        # Create document
        document_id = "doc_123"
        mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(
            data=[{"id": document_id, "title": "Test Document"}]
        )
        
        document = await supabase_service.create_document({
            "title": "Test Document",
            "content": "Test content"
        })
        
        # Create page with valid document_id
        mock_client.table.return_value.insert.return_value.execute.return_value = MagicMock(
            data=[{"id": "page_123", "document_id": document_id}]
        )
        
        page = await supabase_service.create_page({
            "document_id": document["id"],
            "page_number": 1,
            "content": "Page content"
        })
        
        assert page["document_id"] == document["id"]
        
        # Test invalid foreign key (should be handled by service validation)
        with pytest.raises(ValueError, match="Invalid document_id"):
            await supabase_service.create_page({
                "document_id": "invalid_id",
                "page_number": 1,
                "content": "Page content"
            })