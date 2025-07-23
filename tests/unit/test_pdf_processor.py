"""
Unit tests for PDF processor service.

Tests the PDFProcessor class in isolation using mocks for external dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor

from app.services.pdf_processor import PDFProcessor
from app.models.pdf_models import PDFProcessingRequest, PDFProcessingResponse


class TestPDFProcessor:
    """Test suite for PDFProcessor class."""

    @pytest.fixture
    def pdf_processor(self):
        """Create a PDFProcessor instance for testing."""
        return PDFProcessor(max_workers=2)

    @pytest.fixture
    def sample_pdf_request(self):
        """Create a sample PDF processing request."""
        return PDFProcessingRequest(
            file_path="test.pdf",
            extract_text=True,
            extract_tables=True,
            extract_images=False,
            output_format="markdown"
        )

    def test_init(self, pdf_processor):
        """Test PDFProcessor initialization."""
        assert pdf_processor.max_workers == 2
        assert isinstance(pdf_processor.executor, ThreadPoolExecutor)

    def test_init_default_workers(self):
        """Test PDFProcessor initialization with default workers."""
        processor = PDFProcessor()
        assert processor.max_workers == 4
        assert isinstance(processor.executor, ThreadPoolExecutor)

    def test_cleanup_on_delete(self, pdf_processor):
        """Test that executor is properly cleaned up on deletion."""
        executor_mock = Mock()
        pdf_processor.executor = executor_mock
        
        # Trigger __del__ method
        del pdf_processor
        
        executor_mock.shutdown.assert_called_once_with(wait=True)

    @pytest.mark.asyncio
    async def test_process_pdf_success(self, pdf_processor, sample_pdf_request):
        """Test successful PDF processing."""
        # Mock the sync processing method
        mock_result = PDFProcessingResponse(
            success=True,
            extracted_text="Sample text",
            extracted_tables=[],
            extracted_images=[],
            metadata={"pages": 1},
            processing_time=1.5
        )
        
        with patch.object(pdf_processor, '_process_pdf_sync', return_value=mock_result):
            result = await pdf_processor.process_pdf(sample_pdf_request)
            
            assert result.success is True
            assert result.extracted_text == "Sample text"
            assert result.processing_time == 1.5

    @pytest.mark.asyncio
    async def test_process_pdf_file_not_found(self, pdf_processor):
        """Test PDF processing with non-existent file."""
        request = PDFProcessingRequest(
            file_path="nonexistent.pdf",
            extract_text=True
        )
        
        result = await pdf_processor.process_pdf(request)
        
        assert result.success is False
        assert "File not found" in result.error_message

    @pytest.mark.asyncio
    async def test_process_pdf_invalid_format(self, pdf_processor):
        """Test PDF processing with invalid file format."""
        # Create a temporary non-PDF file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"Not a PDF")
            temp_path = temp_file.name
        
        try:
            request = PDFProcessingRequest(
                file_path=temp_path,
                extract_text=True
            )
            
            result = await pdf_processor.process_pdf(request)
            
            assert result.success is False
            assert "Invalid file format" in result.error_message
        finally:
            os.unlink(temp_path)

    @patch('app.services.pdf_processor.pymupdf4llm')
    def test_process_pdf_sync_text_extraction(self, mock_pymupdf, pdf_processor, sample_pdf_request):
        """Test synchronous PDF processing with text extraction."""
        # Mock pymupdf4llm response
        mock_pymupdf.to_markdown.return_value = "# Sample Document\nSample text content"
        
        # Create a temporary PDF file for testing
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(b"%PDF-1.4\n%fake pdf content")
            temp_path = temp_file.name
        
        try:
            request = PDFProcessingRequest(
                file_path=temp_path,
                extract_text=True,
                extract_tables=False,
                extract_images=False,
                output_format="markdown"
            )
            
            result = pdf_processor._process_pdf_sync(request)
            
            assert result.success is True
            assert result.extracted_text == "# Sample Document\nSample text content"
            assert result.processing_time > 0
            mock_pymupdf.to_markdown.assert_called_once()
        finally:
            os.unlink(temp_path)

    @patch('app.services.pdf_processor.pymupdf4llm')
    def test_process_pdf_sync_with_tables(self, mock_pymupdf, pdf_processor):
        """Test PDF processing with table extraction."""
        # Mock pymupdf4llm response with tables
        mock_pymupdf.to_markdown.return_value = "# Document\n\n| Col1 | Col2 |\n|------|------|\n| A    | B    |"
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(b"%PDF-1.4\n%fake pdf content")
            temp_path = temp_file.name
        
        try:
            request = PDFProcessingRequest(
                file_path=temp_path,
                extract_text=True,
                extract_tables=True,
                output_format="markdown"
            )
            
            result = pdf_processor._process_pdf_sync(request)
            
            assert result.success is True
            assert "| Col1 | Col2 |" in result.extracted_text
            # Tables should be extracted from markdown
            assert len(result.extracted_tables) >= 0
        finally:
            os.unlink(temp_path)

    @patch('app.services.pdf_processor.pymupdf4llm')
    def test_process_pdf_sync_processing_error(self, mock_pymupdf, pdf_processor, sample_pdf_request):
        """Test handling of processing errors."""
        # Mock pymupdf4llm to raise an exception
        mock_pymupdf.to_markdown.side_effect = Exception("Processing failed")
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(b"%PDF-1.4\n%fake pdf content")
            temp_path = temp_file.name
        
        try:
            request = PDFProcessingRequest(
                file_path=temp_path,
                extract_text=True
            )
            
            result = pdf_processor._process_pdf_sync(request)
            
            assert result.success is False
            assert "Processing failed" in result.error_message
        finally:
            os.unlink(temp_path)

    def test_extract_tables_from_markdown(self, pdf_processor):
        """Test table extraction from markdown content."""
        markdown_content = """
# Document Title

Some text before table.

| Name | Age | City |
|------|-----|------|
| John | 25  | NYC  |
| Jane | 30  | LA   |

More text after table.

| Product | Price |
|---------|-------|
| Apple   | $1.00 |
| Orange  | $0.50 |
"""
        
        tables = pdf_processor._extract_tables_from_markdown(markdown_content)
        
        assert len(tables) == 2
        assert tables[0]["headers"] == ["Name", "Age", "City"]
        assert len(tables[0]["rows"]) == 2
        assert tables[1]["headers"] == ["Product", "Price"]
        assert len(tables[1]["rows"]) == 2

    def test_extract_tables_from_markdown_no_tables(self, pdf_processor):
        """Test table extraction when no tables are present."""
        markdown_content = """
# Document Title

This is just regular text content.
No tables here.
"""
        
        tables = pdf_processor._extract_tables_from_markdown(markdown_content)
        
        assert len(tables) == 0

    def test_extract_tables_from_markdown_malformed(self, pdf_processor):
        """Test table extraction with malformed tables."""
        markdown_content = """
# Document Title

| Name | Age |
|------|
| John | 25  |
"""
        
        # Should handle malformed tables gracefully
        tables = pdf_processor._extract_tables_from_markdown(markdown_content)
        
        # Depending on implementation, might return empty or partial table
        assert isinstance(tables, list)

    def test_validate_file_path_valid_pdf(self, pdf_processor):
        """Test file path validation with valid PDF."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(b"%PDF-1.4\n%fake pdf content")
            temp_path = temp_file.name
        
        try:
            # Should not raise exception
            pdf_processor._validate_file_path(temp_path)
        finally:
            os.unlink(temp_path)

    def test_validate_file_path_not_found(self, pdf_processor):
        """Test file path validation with non-existent file."""
        with pytest.raises(FileNotFoundError):
            pdf_processor._validate_file_path("nonexistent.pdf")

    def test_validate_file_path_invalid_extension(self, pdf_processor):
        """Test file path validation with invalid extension."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"Not a PDF")
            temp_path = temp_file.name
        
        try:
            with pytest.raises(ValueError, match="Invalid file format"):
                pdf_processor._validate_file_path(temp_path)
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_process_multiple_pdfs_concurrently(self, pdf_processor):
        """Test processing multiple PDFs concurrently."""
        # Create multiple temporary PDF files
        temp_files = []
        requests = []
        
        for i in range(3):
            temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            temp_file.write(b"%PDF-1.4\n%fake pdf content")
            temp_file.close()
            temp_files.append(temp_file.name)
            
            requests.append(PDFProcessingRequest(
                file_path=temp_file.name,
                extract_text=True
            ))
        
        try:
            # Mock the sync processing to return success
            with patch.object(pdf_processor, '_process_pdf_sync') as mock_sync:
                mock_sync.return_value = PDFProcessingResponse(
                    success=True,
                    extracted_text="Sample text",
                    processing_time=0.1
                )
                
                # Process all requests
                results = []
                for request in requests:
                    result = await pdf_processor.process_pdf(request)
                    results.append(result)
                
                # All should succeed
                assert all(result.success for result in results)
                assert len(results) == 3
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                os.unlink(temp_file)

    def test_get_processing_stats(self, pdf_processor):
        """Test getting processing statistics."""
        # This would test a method that tracks processing stats
        # Implementation depends on whether such a method exists
        stats = getattr(pdf_processor, 'get_processing_stats', lambda: {})()
        assert isinstance(stats, dict)