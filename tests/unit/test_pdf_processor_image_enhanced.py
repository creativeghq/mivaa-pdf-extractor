"""
Enhanced unit tests for PDF processor image extraction functionality.

Tests the new image processing capabilities added to PDFProcessor including:
- Image extraction with various PDF types
- Image format conversion
- Metadata extraction
- Quality assessment
- Duplicate removal
- Enhanced image processing features
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock, mock_open
from pathlib import Path
import tempfile
import os
import io
from PIL import Image as PILImage
import numpy as np
from typing import Dict, Any, List

from app.services.pdf_processor import PDFProcessor
from app.models.pdf_models import PDFProcessingRequest, PDFProcessingResponse


class TestPDFProcessorImageEnhanced:
    """Enhanced test suite for PDFProcessor image processing capabilities."""

    @pytest.fixture
    def pdf_processor(self):
        """Create a PDFProcessor instance for testing."""
        return PDFProcessor(max_workers=2)

    @pytest.fixture
    def sample_image_data(self):
        """Create sample image data for testing."""
        # Create different types of test images
        images = {}
        
        # High quality image
        img_hq = PILImage.new('RGB', (800, 600), color='red')
        img_hq_bytes = io.BytesIO()
        img_hq.save(img_hq_bytes, format='PNG')
        img_hq_bytes.seek(0)
        images['high_quality'] = img_hq_bytes.getvalue()
        
        # Low quality image
        img_lq = PILImage.new('RGB', (100, 75), color='blue')
        img_lq_bytes = io.BytesIO()
        img_lq.save(img_lq_bytes, format='JPEG', quality=30)
        img_lq_bytes.seek(0)
        images['low_quality'] = img_lq_bytes.getvalue()
        
        # Duplicate image (same as high quality)
        images['duplicate'] = images['high_quality']
        
        # Different format image
        img_png = PILImage.new('RGB', (400, 300), color='green')
        img_png_bytes = io.BytesIO()
        img_png.save(img_png_bytes, format='PNG')
        img_png_bytes.seek(0)
        images['png_format'] = img_png_bytes.getvalue()
        
        return images

    @pytest.fixture
    def mock_pdf_with_images(self):
        """Mock PDF document with images."""
        mock_doc = Mock()
        mock_page = Mock()
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_doc.page_count = 1
        
        # Mock image list
        mock_image_list = [
            {
                'xref': 1,
                'smask': 0,
                'width': 800,
                'height': 600,
                'colorspace': 3,
                'bpc': 8,
                'ext': 'png',
                'cs-name': 'DeviceRGB'
            },
            {
                'xref': 2,
                'smask': 0,
                'width': 400,
                'height': 300,
                'colorspace': 3,
                'bpc': 8,
                'ext': 'jpeg',
                'cs-name': 'DeviceRGB'
            }
        ]
        mock_page.get_images.return_value = mock_image_list
        
        return mock_doc

    def test_extract_images_sync_success(self, pdf_processor, sample_image_data, mock_pdf_with_images):
        """Test successful synchronous image extraction."""
        with patch('fitz.open', return_value=mock_pdf_with_images):
            with patch.object(mock_pdf_with_images, 'extract_image') as mock_extract:
                # Mock extracted image data
                mock_extract.side_effect = [
                    {'image': sample_image_data['high_quality'], 'ext': 'png'},
                    {'image': sample_image_data['png_format'], 'ext': 'jpeg'}
                ]
                
                # Create temporary PDF file
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                    temp_file.write(b"%PDF-1.4\n%fake pdf content")
                    temp_path = temp_file.name
                
                try:
                    result = pdf_processor._extract_images_sync(temp_path)
                    
                    assert result['success'] is True
                    assert len(result['images']) == 2
                    assert result['total_images'] == 2
                    assert result['processing_time'] > 0
                    
                    # Check first image
                    img1 = result['images'][0]
                    assert img1['width'] == 800
                    assert img1['height'] == 600
                    assert img1['format'] == 'PNG'
                    assert 'data' in img1
                    assert 'metadata' in img1
                    
                finally:
                    os.unlink(temp_path)

    def test_extract_images_sync_with_processing_options(self, pdf_processor, sample_image_data, mock_pdf_with_images):
        """Test image extraction with various processing options."""
        processing_options = {
            'convert_format': 'JPEG',
            'enhance_quality': True,
            'extract_metadata': True,
            'remove_duplicates': True,
            'min_size': (100, 100),
            'max_size': (2000, 2000)
        }
        
        with patch('fitz.open', return_value=mock_pdf_with_images):
            with patch.object(mock_pdf_with_images, 'extract_image') as mock_extract:
                mock_extract.side_effect = [
                    {'image': sample_image_data['high_quality'], 'ext': 'png'},
                    {'image': sample_image_data['duplicate'], 'ext': 'png'}  # Duplicate
                ]
                
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                    temp_file.write(b"%PDF-1.4\n%fake pdf content")
                    temp_path = temp_file.name
                
                try:
                    result = pdf_processor._extract_images_sync(temp_path, processing_options)
                    
                    assert result['success'] is True
                    # Should have only 1 image after duplicate removal
                    assert len(result['images']) == 1
                    assert result['duplicates_removed'] == 1
                    
                    # Check converted format
                    img = result['images'][0]
                    assert img['format'] == 'JPEG'
                    assert 'enhanced' in img
                    assert img['enhanced'] is True
                    
                finally:
                    os.unlink(temp_path)

    def test_process_extracted_image_format_conversion(self, pdf_processor, sample_image_data):
        """Test image format conversion during processing."""
        image_data = sample_image_data['png_format']
        image_info = {
            'xref': 1,
            'width': 400,
            'height': 300,
            'ext': 'png'
        }
        
        options = {'convert_format': 'JPEG'}
        
        result = pdf_processor._process_extracted_image(image_data, image_info, options)
        
        assert result['success'] is True
        assert result['format'] == 'JPEG'
        assert result['original_format'] == 'PNG'
        assert 'data' in result
        assert result['size'] == (400, 300)

    def test_process_extracted_image_quality_enhancement(self, pdf_processor, sample_image_data):
        """Test image quality enhancement during processing."""
        image_data = sample_image_data['low_quality']
        image_info = {
            'xref': 1,
            'width': 100,
            'height': 75,
            'ext': 'jpeg'
        }
        
        options = {'enhance_quality': True}
        
        with patch('cv2.imread') as mock_imread, \
             patch('cv2.bilateralFilter') as mock_filter, \
             patch('cv2.imencode') as mock_encode:
            
            # Mock OpenCV operations
            mock_img = np.zeros((75, 100, 3), dtype=np.uint8)
            mock_imread.return_value = mock_img
            mock_filter.return_value = mock_img
            mock_encode.return_value = (True, mock_img.tobytes())
            
            result = pdf_processor._process_extracted_image(image_data, image_info, options)
            
            assert result['success'] is True
            assert result['enhanced'] is True
            assert 'enhancement_applied' in result

    def test_calculate_image_quality(self, pdf_processor, sample_image_data):
        """Test image quality calculation."""
        # Test high quality image
        hq_score = pdf_processor._calculate_image_quality(sample_image_data['high_quality'])
        assert 0.0 <= hq_score <= 1.0
        assert hq_score > 0.5  # Should be relatively high quality
        
        # Test low quality image
        lq_score = pdf_processor._calculate_image_quality(sample_image_data['low_quality'])
        assert 0.0 <= lq_score <= 1.0
        assert lq_score < hq_score  # Should be lower quality

    def test_calculate_image_quality_invalid_data(self, pdf_processor):
        """Test image quality calculation with invalid data."""
        invalid_data = b"not an image"
        score = pdf_processor._calculate_image_quality(invalid_data)
        assert score == 0.0

    def test_remove_duplicate_images(self, pdf_processor, sample_image_data):
        """Test duplicate image removal."""
        images = [
            {
                'id': 'img1',
                'data': sample_image_data['high_quality'],
                'hash': 'hash1',
                'size': (800, 600)
            },
            {
                'id': 'img2',
                'data': sample_image_data['duplicate'],  # Same as high_quality
                'hash': 'hash1',  # Same hash
                'size': (800, 600)
            },
            {
                'id': 'img3',
                'data': sample_image_data['png_format'],
                'hash': 'hash2',
                'size': (400, 300)
            }
        ]
        
        unique_images, removed_count = pdf_processor._remove_duplicate_images(images)
        
        assert len(unique_images) == 2
        assert removed_count == 1
        assert unique_images[0]['id'] == 'img1'  # First occurrence kept
        assert unique_images[1]['id'] == 'img3'

    def test_extract_image_metadata(self, pdf_processor, sample_image_data):
        """Test image metadata extraction."""
        # Create image with EXIF data
        img = PILImage.new('RGB', (800, 600), color='red')
        
        # Add some metadata
        exif_dict = {
            'Exif': {
                'DateTime': '2024:03:15 10:30:00',
                'Software': 'Test Software'
            }
        }
        
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG', exif=img.getexif())
        img_bytes.seek(0)
        
        metadata = pdf_processor._extract_image_metadata(img_bytes.getvalue())
        
        assert 'width' in metadata
        assert 'height' in metadata
        assert 'format' in metadata
        assert 'mode' in metadata
        assert metadata['width'] == 800
        assert metadata['height'] == 600

    def test_extract_image_metadata_no_exif(self, pdf_processor, sample_image_data):
        """Test metadata extraction for image without EXIF data."""
        metadata = pdf_processor._extract_image_metadata(sample_image_data['png_format'])
        
        assert 'width' in metadata
        assert 'height' in metadata
        assert 'format' in metadata
        assert metadata['format'] == 'PNG'

    @pytest.mark.asyncio
    async def test_process_pdf_with_image_extraction_enabled(self, pdf_processor, mock_pdf_with_images, sample_image_data):
        """Test PDF processing with image extraction enabled."""
        request = PDFProcessingRequest(
            file_path="test.pdf",
            extract_text=True,
            extract_images=True,
            image_processing_options={
                'convert_format': 'JPEG',
                'extract_metadata': True,
                'remove_duplicates': True
            }
        )
        
        with patch('fitz.open', return_value=mock_pdf_with_images):
            with patch.object(mock_pdf_with_images, 'extract_image') as mock_extract:
                mock_extract.side_effect = [
                    {'image': sample_image_data['high_quality'], 'ext': 'png'},
                    {'image': sample_image_data['png_format'], 'ext': 'jpeg'}
                ]
                
                with patch('pymupdf4llm.to_markdown', return_value="# Test Document\nSample text"):
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                        temp_file.write(b"%PDF-1.4\n%fake pdf content")
                        temp_path = temp_file.name
                        request.file_path = temp_path
                    
                    try:
                        result = await pdf_processor.process_pdf(request)
                        
                        assert result.success is True
                        assert len(result.extracted_images) == 2
                        assert result.extracted_text == "# Test Document\nSample text"
                        
                        # Check image processing results
                        for img in result.extracted_images:
                            assert 'format' in img
                            assert 'metadata' in img
                            assert 'quality_score' in img
                            
                    finally:
                        os.unlink(temp_path)

    def test_image_size_filtering(self, pdf_processor, sample_image_data, mock_pdf_with_images):
        """Test filtering images by size constraints."""
        processing_options = {
            'min_size': (200, 200),  # Minimum size filter
            'max_size': (1000, 1000)  # Maximum size filter
        }
        
        with patch('fitz.open', return_value=mock_pdf_with_images):
            with patch.object(mock_pdf_with_images, 'extract_image') as mock_extract:
                # Mock images of different sizes
                mock_extract.side_effect = [
                    {'image': sample_image_data['high_quality'], 'ext': 'png'},  # 800x600 - should pass
                    {'image': sample_image_data['low_quality'], 'ext': 'jpeg'}   # 100x75 - should be filtered
                ]
                
                # Update mock image list to reflect actual sizes
                mock_pdf_with_images.__iter__.return_value[0].get_images.return_value = [
                    {'xref': 1, 'width': 800, 'height': 600, 'ext': 'png'},
                    {'xref': 2, 'width': 100, 'height': 75, 'ext': 'jpeg'}
                ]
                
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                    temp_file.write(b"%PDF-1.4\n%fake pdf content")
                    temp_path = temp_file.name
                
                try:
                    result = pdf_processor._extract_images_sync(temp_path, processing_options)
                    
                    assert result['success'] is True
                    assert len(result['images']) == 1  # Only high quality image should pass
                    assert result['images'][0]['width'] == 800
                    assert result['filtered_count'] == 1
                    
                finally:
                    os.unlink(temp_path)

    def test_image_extraction_error_handling(self, pdf_processor, mock_pdf_with_images):
        """Test error handling during image extraction."""
        with patch('fitz.open', return_value=mock_pdf_with_images):
            with patch.object(mock_pdf_with_images, 'extract_image') as mock_extract:
                # Mock extraction error
                mock_extract.side_effect = Exception("Image extraction failed")
                
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                    temp_file.write(b"%PDF-1.4\n%fake pdf content")
                    temp_path = temp_file.name
                
                try:
                    result = pdf_processor._extract_images_sync(temp_path)
                    
                    assert result['success'] is False
                    assert "Image extraction failed" in result['error']
                    assert result['images'] == []
                    
                finally:
                    os.unlink(temp_path)

    def test_image_processing_with_corrupted_data(self, pdf_processor):
        """Test image processing with corrupted image data."""
        corrupted_data = b"corrupted image data"
        image_info = {
            'xref': 1,
            'width': 100,
            'height': 100,
            'ext': 'png'
        }
        
        result = pdf_processor._process_extracted_image(corrupted_data, image_info, {})
        
        assert result['success'] is False
        assert 'error' in result

    def test_image_hash_calculation(self, pdf_processor, sample_image_data):
        """Test image hash calculation for duplicate detection."""
        # Same image should produce same hash
        hash1 = pdf_processor._calculate_image_hash(sample_image_data['high_quality'])
        hash2 = pdf_processor._calculate_image_hash(sample_image_data['duplicate'])
        
        assert hash1 == hash2
        
        # Different images should produce different hashes
        hash3 = pdf_processor._calculate_image_hash(sample_image_data['png_format'])
        assert hash1 != hash3

    def test_image_enhancement_options(self, pdf_processor, sample_image_data):
        """Test various image enhancement options."""
        image_data = sample_image_data['low_quality']
        image_info = {
            'xref': 1,
            'width': 100,
            'height': 75,
            'ext': 'jpeg'
        }
        
        enhancement_options = {
            'enhance_quality': True,
            'sharpen': True,
            'denoise': True,
            'brightness': 1.1,
            'contrast': 1.2
        }
        
        with patch('cv2.imread') as mock_imread, \
             patch('cv2.bilateralFilter') as mock_filter, \
             patch('cv2.filter2D') as mock_sharpen, \
             patch('cv2.convertScaleAbs') as mock_adjust, \
             patch('cv2.imencode') as mock_encode:
            
            mock_img = np.zeros((75, 100, 3), dtype=np.uint8)
            mock_imread.return_value = mock_img
            mock_filter.return_value = mock_img
            mock_sharpen.return_value = mock_img
            mock_adjust.return_value = mock_img
            mock_encode.return_value = (True, mock_img.tobytes())
            
            result = pdf_processor._process_extracted_image(image_data, image_info, enhancement_options)
            
            assert result['success'] is True
            assert result['enhanced'] is True
            assert 'enhancement_details' in result

    @pytest.mark.asyncio
    async def test_backward_compatibility(self, pdf_processor):
        """Test that enhanced image processing maintains backward compatibility."""
        # Test with old-style request (no image processing options)
        request = PDFProcessingRequest(
            file_path="test.pdf",
            extract_text=True,
            extract_images=False  # Images disabled
        )
        
        with patch('pymupdf4llm.to_markdown', return_value="Sample text"):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_file.write(b"%PDF-1.4\n%fake pdf content")
                temp_path = temp_file.name
                request.file_path = temp_path
            
            try:
                result = await pdf_processor.process_pdf(request)
                
                assert result.success is True
                assert result.extracted_text == "Sample text"
                assert result.extracted_images == []  # No images extracted
                
            finally:
                os.unlink(temp_path)

    def test_performance_metrics_collection(self, pdf_processor, sample_image_data, mock_pdf_with_images):
        """Test that performance metrics are collected during image processing."""
        with patch('fitz.open', return_value=mock_pdf_with_images):
            with patch.object(mock_pdf_with_images, 'extract_image') as mock_extract:
                mock_extract.side_effect = [
                    {'image': sample_image_data['high_quality'], 'ext': 'png'}
                ]
                
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                    temp_file.write(b"%PDF-1.4\n%fake pdf content")
                    temp_path = temp_file.name
                
                try:
                    result = pdf_processor._extract_images_sync(temp_path)
                    
                    assert result['success'] is True
                    assert 'processing_time' in result
                    assert 'performance_metrics' in result
                    assert result['processing_time'] > 0
                    
                    metrics = result['performance_metrics']
                    assert 'extraction_time' in metrics
                    assert 'processing_time' in metrics
                    assert 'total_images' in metrics
                    
                finally:
                    os.unlink(temp_path)