"""
Integration tests for PDF processing with image extraction.

Tests end-to-end functionality including:
- Processing different PDF types with images
- Integration between PDFProcessor and ImageProcessor
- Real file processing workflows
- Performance with various PDF sizes and image counts
"""

import pytest
import asyncio
import tempfile
import os
import shutil
from pathlib import Path
from typing import Dict, List
import time

from app.services.pdf_processor import PDFProcessor
from app.models.pdf_models import PDFProcessingRequest, PDFProcessingResponse


class TestPDFImageIntegration:
    """Integration test suite for PDF image processing."""

    @pytest.fixture(scope="class")
    def test_data_dir(self):
        """Get the test data directory."""
        return Path(__file__).parent.parent / "data"

    @pytest.fixture(scope="class")
    def pdf_processor(self):
        """Create a PDFProcessor instance for integration testing."""
        return PDFProcessor(max_workers=4)

    @pytest.fixture(scope="class")
    def sample_pdfs(self, test_data_dir):
        """Create sample PDF files for testing if they don't exist."""
        # First, try to create test PDFs using our script
        create_script = test_data_dir / "create_test_pdfs.py"
        if create_script.exists():
            try:
                import subprocess
                import sys
                result = subprocess.run([sys.executable, str(create_script)], 
                                      capture_output=True, text=True, cwd=str(test_data_dir))
                if result.returncode != 0:
                    print(f"Warning: Could not create test PDFs: {result.stderr}")
            except Exception as e:
                print(f"Warning: Could not run PDF creation script: {e}")
        
        # Return expected PDF paths
        return {
            'text_heavy': test_data_dir / "text_heavy" / "sample_text_heavy.pdf",
            'image_heavy': test_data_dir / "image_heavy" / "sample_image_heavy.pdf",
            'mixed_content': test_data_dir / "mixed_content" / "sample_mixed_content.pdf",
            'scanned_doc': test_data_dir / "scanned_docs" / "sample_scanned_doc.pdf"
        }

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_text_heavy_pdf_processing(self, pdf_processor, sample_pdfs):
        """Test processing text-heavy PDF with minimal images."""
        pdf_path = sample_pdfs['text_heavy']
        
        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")
        
        request = PDFProcessingRequest(
            file_path=str(pdf_path),
            extract_text=True,
            extract_images=True,
            extract_tables=True,
            image_processing_options={
                'extract_metadata': True,
                'min_size': (50, 50)
            }
        )
        
        start_time = time.time()
        result = await pdf_processor.process_pdf(request)
        processing_time = time.time() - start_time
        
        # Assertions
        assert result.success is True
        assert result.extracted_text is not None
        assert len(result.extracted_text) > 100  # Should have substantial text
        assert isinstance(result.extracted_images, list)
        assert len(result.extracted_images) <= 2  # Text-heavy should have few images
        assert result.processing_time > 0
        assert processing_time < 30  # Should complete within reasonable time
        
        # Check image metadata if images were extracted
        for image in result.extracted_images:
            assert 'metadata' in image
            assert 'width' in image['metadata']
            assert 'height' in image['metadata']

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_image_heavy_pdf_processing(self, pdf_processor, sample_pdfs):
        """Test processing image-heavy PDF with multiple images."""
        pdf_path = sample_pdfs['image_heavy']
        
        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")
        
        request = PDFProcessingRequest(
            file_path=str(pdf_path),
            extract_text=True,
            extract_images=True,
            image_processing_options={
                'convert_format': 'JPEG',
                'extract_metadata': True,
                'remove_duplicates': True,
                'enhance_quality': False  # Skip enhancement for faster testing
            }
        )
        
        start_time = time.time()
        result = await pdf_processor.process_pdf(request)
        processing_time = time.time() - start_time
        
        # Assertions
        assert result.success is True
        assert result.extracted_text is not None
        assert isinstance(result.extracted_images, list)
        assert len(result.extracted_images) >= 2  # Image-heavy should have multiple images
        assert processing_time < 60  # Should complete within reasonable time
        
        # Check image processing results
        for image in result.extracted_images:
            assert 'format' in image
            assert image['format'] == 'JPEG'  # Should be converted
            assert 'metadata' in image
            assert 'quality_score' in image
            assert 0 <= image['quality_score'] <= 1

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_mixed_content_pdf_processing(self, pdf_processor, sample_pdfs):
        """Test processing mixed content PDF with balanced text and images."""
        pdf_path = sample_pdfs['mixed_content']
        
        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")
        
        request = PDFProcessingRequest(
            file_path=str(pdf_path),
            extract_text=True,
            extract_images=True,
            extract_tables=True,
            image_processing_options={
                'extract_metadata': True,
                'remove_duplicates': True,
                'min_size': (100, 100)
            }
        )
        
        result = await pdf_processor.process_pdf(request)
        
        # Assertions
        assert result.success is True
        assert result.extracted_text is not None
        assert len(result.extracted_text) > 50  # Should have reasonable text
        assert isinstance(result.extracted_images, list)
        assert isinstance(result.extracted_tables, list)
        
        # Mixed content should have both text and images
        assert len(result.extracted_text.strip()) > 0
        assert len(result.extracted_images) > 0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_scanned_document_processing(self, pdf_processor, sample_pdfs):
        """Test processing scanned document PDF."""
        pdf_path = sample_pdfs['scanned_doc']
        
        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")
        
        request = PDFProcessingRequest(
            file_path=str(pdf_path),
            extract_text=True,
            extract_images=True,
            image_processing_options={
                'enhance_quality': True,  # Important for scanned docs
                'extract_metadata': True
            }
        )
        
        result = await pdf_processor.process_pdf(request)
        
        # Assertions
        assert result.success is True
        # Scanned documents might have limited text extraction
        assert result.extracted_text is not None
        assert isinstance(result.extracted_images, list)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_pdf_processing(self, pdf_processor, sample_pdfs):
        """Test processing multiple PDFs concurrently."""
        # Get available PDF files
        available_pdfs = [path for path in sample_pdfs.values() if path.exists()]
        
        if len(available_pdfs) < 2:
            pytest.skip("Need at least 2 test PDFs for concurrent testing")
        
        # Create requests for concurrent processing
        requests = []
        for i, pdf_path in enumerate(available_pdfs[:3]):  # Limit to 3 for testing
            request = PDFProcessingRequest(
                file_path=str(pdf_path),
                extract_text=True,
                extract_images=True,
                image_processing_options={
                    'extract_metadata': True,
                    'enhance_quality': False  # Skip for faster testing
                }
            )
            requests.append(request)
        
        # Process concurrently
        start_time = time.time()
        tasks = [pdf_processor.process_pdf(req) for req in requests]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Assertions
        assert len(results) == len(requests)
        for result in results:
            assert result.success is True
            assert result.extracted_text is not None
        
        # Concurrent processing should be faster than sequential
        # (This is a rough check - actual speedup depends on system)
        assert total_time < sum(r.processing_time for r in results) * 0.8

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_image_format_conversion_integration(self, pdf_processor, sample_pdfs):
        """Test end-to-end image format conversion."""
        # Use image-heavy PDF for this test
        pdf_path = sample_pdfs['image_heavy']
        
        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")
        
        # Test PNG conversion
        request_png = PDFProcessingRequest(
            file_path=str(pdf_path),
            extract_text=False,  # Focus on images
            extract_images=True,
            image_processing_options={
                'convert_format': 'PNG',
                'extract_metadata': True
            }
        )
        
        result_png = await pdf_processor.process_pdf(request_png)
        
        # Test JPEG conversion
        request_jpeg = PDFProcessingRequest(
            file_path=str(pdf_path),
            extract_text=False,
            extract_images=True,
            image_processing_options={
                'convert_format': 'JPEG',
                'extract_metadata': True
            }
        )
        
        result_jpeg = await pdf_processor.process_pdf(request_jpeg)
        
        # Assertions
        assert result_png.success is True
        assert result_jpeg.success is True
        
        # Check format conversion
        for image in result_png.extracted_images:
            assert image['format'] == 'PNG'
        
        for image in result_jpeg.extracted_images:
            assert image['format'] == 'JPEG'

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_duplicate_removal_integration(self, pdf_processor, sample_pdfs):
        """Test duplicate image removal in real PDF processing."""
        pdf_path = sample_pdfs['image_heavy']
        
        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")
        
        # Process with duplicate removal enabled
        request_with_dedup = PDFProcessingRequest(
            file_path=str(pdf_path),
            extract_text=False,
            extract_images=True,
            image_processing_options={
                'remove_duplicates': True,
                'extract_metadata': True
            }
        )
        
        # Process without duplicate removal
        request_without_dedup = PDFProcessingRequest(
            file_path=str(pdf_path),
            extract_text=False,
            extract_images=True,
            image_processing_options={
                'remove_duplicates': False,
                'extract_metadata': True
            }
        )
        
        result_with_dedup = await pdf_processor.process_pdf(request_with_dedup)
        result_without_dedup = await pdf_processor.process_pdf(request_without_dedup)
        
        # Assertions
        assert result_with_dedup.success is True
        assert result_without_dedup.success is True
        
        # With deduplication should have same or fewer images
        assert len(result_with_dedup.extracted_images) <= len(result_without_dedup.extracted_images)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_size_filtering_integration(self, pdf_processor, sample_pdfs):
        """Test image size filtering in real PDF processing."""
        pdf_path = sample_pdfs['image_heavy']
        
        if not pdf_path.exists():
            pytest.skip(f"Test PDF not found: {pdf_path}")
        
        # Process with strict size filtering
        request_strict = PDFProcessingRequest(
            file_path=str(pdf_path),
            extract_text=False,
            extract_images=True,
            image_processing_options={
                'min_size': (300, 300),  # Strict minimum
                'max_size': (1000, 1000),
                'extract_metadata': True
            }
        )
        
        # Process with lenient size filtering
        request_lenient = PDFProcessingRequest(
            file_path=str(pdf_path),
            extract_text=False,
            extract_images=True,
            image_processing_options={
                'min_size': (50, 50),  # Lenient minimum
                'max_size': (2000, 2000),
                'extract_metadata': True
            }
        )
        
        result_strict = await pdf_processor.process_pdf(request_strict)
        result_lenient = await pdf_processor.process_pdf(request_lenient)
        
        # Assertions
        assert result_strict.success is True
        assert result_lenient.success is True
        
        # Strict filtering should result in fewer images
        assert len(result_strict.extracted_images) <= len(result_lenient.extracted_images)
        
        # Check that remaining images meet size requirements
        for image in result_strict.extracted_images:
            metadata = image['metadata']
            assert metadata['width'] >= 300
            assert metadata['height'] >= 300
            assert metadata['width'] <= 1000
            assert metadata['height'] <= 1000

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_handling_integration(self, pdf_processor):
        """Test error handling with invalid files."""
        # Test with non-existent file
        request_missing = PDFProcessingRequest(
            file_path="nonexistent.pdf",
            extract_text=True,
            extract_images=True
        )
        
        result_missing = await pdf_processor.process_pdf(request_missing)
        assert result_missing.success is False
        assert "not found" in result_missing.error_message.lower()
        
        # Test with invalid file format
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"This is not a PDF file")
            temp_path = temp_file.name
        
        try:
            request_invalid = PDFProcessingRequest(
                file_path=temp_path,
                extract_text=True,
                extract_images=True
            )
            
            result_invalid = await pdf_processor.process_pdf(request_invalid)
            assert result_invalid.success is False
            assert "invalid" in result_invalid.error_message.lower()
            
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_performance_benchmarking(self, pdf_processor, sample_pdfs):
        """Test performance with different PDF types."""
        performance_results = {}
        
        for pdf_type, pdf_path in sample_pdfs.items():
            if not pdf_path.exists():
                continue
            
            request = PDFProcessingRequest(
                file_path=str(pdf_path),
                extract_text=True,
                extract_images=True,
                image_processing_options={
                    'extract_metadata': True,
                    'enhance_quality': False  # Skip for consistent timing
                }
            )
            
            # Run multiple times for average
            times = []
            for _ in range(3):
                start_time = time.time()
                result = await pdf_processor.process_pdf(request)
                end_time = time.time()
                
                if result.success:
                    times.append(end_time - start_time)
            
            if times:
                performance_results[pdf_type] = {
                    'avg_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times)
                }
        
        # Log performance results for analysis
        print("\nPerformance Results:")
        for pdf_type, metrics in performance_results.items():
            print(f"{pdf_type}: avg={metrics['avg_time']:.2f}s, "
                  f"min={metrics['min_time']:.2f}s, max={metrics['max_time']:.2f}s")
        
        # Basic performance assertions
        for pdf_type, metrics in performance_results.items():
            assert metrics['avg_time'] < 60  # Should complete within 1 minute
            assert metrics['min_time'] > 0   # Should take some time

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_memory_usage_integration(self, pdf_processor, sample_pdfs):
        """Test memory usage during PDF processing."""
        import psutil
        import gc
        
        # Get available PDF
        available_pdf = None
        for pdf_path in sample_pdfs.values():
            if pdf_path.exists():
                available_pdf = pdf_path
                break
        
        if not available_pdf:
            pytest.skip("No test PDFs available")
        
        # Measure memory before processing
        process = psutil.Process()
        gc.collect()  # Force garbage collection
        memory_before = process.memory_info().rss
        
        # Process PDF
        request = PDFProcessingRequest(
            file_path=str(available_pdf),
            extract_text=True,
            extract_images=True,
            image_processing_options={
                'extract_metadata': True,
                'enhance_quality': False
            }
        )
        
        result = await pdf_processor.process_pdf(request)
        
        # Measure memory after processing
        gc.collect()  # Force garbage collection
        memory_after = process.memory_info().rss
        
        memory_increase = memory_after - memory_before
        memory_increase_mb = memory_increase / (1024 * 1024)
        
        # Assertions
        assert result.success is True
        # Memory increase should be reasonable (less than 100MB for test files)
        assert memory_increase_mb < 100
        
        print(f"\nMemory usage: {memory_increase_mb:.2f} MB increase")

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_backward_compatibility_integration(self, pdf_processor, sample_pdfs):
        """Test that new image processing features don't break existing functionality."""
        available_pdf = None
        for pdf_path in sample_pdfs.values():
            if pdf_path.exists():
                available_pdf = pdf_path
                break
        
        if not available_pdf:
            pytest.skip("No test PDFs available")
        
        # Test old-style request (no image processing options)
        old_request = PDFProcessingRequest(
            file_path=str(available_pdf),
            extract_text=True,
            extract_tables=True,
            extract_images=False  # Old behavior
        )
        
        old_result = await pdf_processor.process_pdf(old_request)
        
        # Test new-style request with image processing
        new_request = PDFProcessingRequest(
            file_path=str(available_pdf),
            extract_text=True,
            extract_tables=True,
            extract_images=True,
            image_processing_options={
                'extract_metadata': True
            }
        )
        
        new_result = await pdf_processor.process_pdf(new_request)
        
        # Assertions
        assert old_result.success is True
        assert new_result.success is True
        
        # Text extraction should be consistent
        assert old_result.extracted_text == new_result.extracted_text
        
        # Old request should have no images, new request should have images
        assert len(old_result.extracted_images) == 0
        assert len(new_result.extracted_images) >= 0  # May or may not have images