"""
Unit tests for Image Processing service.

Tests the ImageProcessor class in isolation using mocks for external dependencies.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from typing import Dict, Any, List, Optional
import io
from PIL import Image
import numpy as np

from app.services.image_processor import ImageProcessor


class TestImageProcessor:
    """Test suite for ImageProcessor class."""

    @pytest.fixture
    def mock_material_kai_service(self):
        """Create a mocked Material Kai service."""
        mock_service = AsyncMock()
        
        # Mock successful analysis response
        mock_service.analyze_image.return_value = {
            "success": True,
            "analysis_id": "analysis_123",
            "objects": [
                {"label": "document", "confidence": 0.95, "bbox": [10, 10, 100, 100]},
                {"label": "text", "confidence": 0.87, "bbox": [20, 20, 80, 80]}
            ],
            "extracted_text": "Sample extracted text",
            "faces": [],
            "metadata": {"processing_time": 1.2}
        }
        
        mock_service.batch_analyze_images.return_value = {
            "success": True,
            "batch_id": "batch_456",
            "total_images": 3
        }
        
        mock_service.search_similar_images.return_value = {
            "success": True,
            "search_id": "search_789",
            "results": [
                {"image_id": "img_1", "similarity_score": 0.95},
                {"image_id": "img_2", "similarity_score": 0.87}
            ]
        }
        
        return mock_service

    @pytest.fixture
    def mock_ocr_engine(self):
        """Create a mocked OCR engine."""
        mock_ocr = Mock()
        mock_ocr.extract_text.return_value = {
            "text": "Extracted text from OCR",
            "confidence": 0.92,
            "words": [
                {"text": "Extracted", "confidence": 0.95, "bbox": [10, 10, 50, 20]},
                {"text": "text", "confidence": 0.90, "bbox": [55, 10, 75, 20]},
                {"text": "from", "confidence": 0.88, "bbox": [80, 10, 100, 20]},
                {"text": "OCR", "confidence": 0.94, "bbox": [105, 10, 125, 20]}
            ]
        }
        return mock_ocr

    @pytest.fixture
    def sample_image_data(self):
        """Create sample image data for testing."""
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='white')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()

    @pytest.fixture
    def image_processor(self, mock_material_kai_service, mock_ocr_engine):
        """Create an ImageProcessor instance with mocked dependencies."""
        with patch('app.services.image_processor.MaterialKaiService', return_value=mock_material_kai_service), \
             patch('app.services.image_processor.OCREngine', return_value=mock_ocr_engine):
            processor = ImageProcessor()
            processor.material_kai_service = mock_material_kai_service
            processor.ocr_engine = mock_ocr_engine
            return processor

    @pytest.mark.asyncio
    async def test_initialization(self, image_processor):
        """Test ImageProcessor initialization."""
        assert image_processor.material_kai_service is not None
        assert image_processor.ocr_engine is not None
        assert hasattr(image_processor, 'supported_formats')
        assert 'png' in image_processor.supported_formats
        assert 'jpg' in image_processor.supported_formats

    @pytest.mark.asyncio
    async def test_validate_image_format_success(self, image_processor, sample_image_data):
        """Test successful image format validation."""
        result = await image_processor.validate_image_format(sample_image_data, 'png')
        
        assert result["valid"] is True
        assert result["format"] == "png"
        assert "width" in result
        assert "height" in result

    @pytest.mark.asyncio
    async def test_validate_image_format_unsupported(self, image_processor):
        """Test validation with unsupported format."""
        fake_data = b"fake image data"
        
        result = await image_processor.validate_image_format(fake_data, 'bmp')
        
        assert result["valid"] is False
        assert "not supported" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_validate_image_format_corrupted(self, image_processor):
        """Test validation with corrupted image data."""
        corrupted_data = b"not an image"
        
        result = await image_processor.validate_image_format(corrupted_data, 'png')
        
        assert result["valid"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_extract_text_ocr_success(self, image_processor, sample_image_data):
        """Test successful OCR text extraction."""
        result = await image_processor.extract_text_ocr(sample_image_data)
        
        assert result["success"] is True
        assert result["text"] == "Extracted text from OCR"
        assert result["confidence"] == 0.92
        assert len(result["words"]) == 4

    @pytest.mark.asyncio
    async def test_extract_text_ocr_failure(self, image_processor, sample_image_data):
        """Test OCR text extraction failure."""
        # Mock OCR failure
        image_processor.ocr_engine.extract_text.side_effect = Exception("OCR failed")
        
        result = await image_processor.extract_text_ocr(sample_image_data)
        
        assert result["success"] is False
        assert "OCR failed" in result["error"]

    @pytest.mark.asyncio
    async def test_analyze_with_material_kai_success(self, image_processor, sample_image_data):
        """Test successful Material Kai analysis."""
        analysis_options = {
            "detect_objects": True,
            "extract_text": True,
            "detect_faces": False
        }
        
        result = await image_processor.analyze_with_material_kai(sample_image_data, analysis_options)
        
        assert result["success"] is True
        assert result["analysis_id"] == "analysis_123"
        assert len(result["objects"]) == 2
        assert result["extracted_text"] == "Sample extracted text"

    @pytest.mark.asyncio
    async def test_analyze_with_material_kai_failure(self, image_processor, sample_image_data):
        """Test Material Kai analysis failure."""
        # Mock analysis failure
        image_processor.material_kai_service.analyze_image.return_value = {
            "success": False,
            "error": "Analysis service unavailable"
        }
        
        result = await image_processor.analyze_with_material_kai(sample_image_data)
        
        assert result["success"] is False
        assert "Analysis service unavailable" in result["error"]

    @pytest.mark.asyncio
    async def test_process_image_comprehensive_success(self, image_processor, sample_image_data):
        """Test comprehensive image processing."""
        processing_options = {
            "extract_text": True,
            "analyze_objects": True,
            "detect_faces": True,
            "enhance_quality": False
        }
        
        result = await image_processor.process_image_comprehensive(
            sample_image_data, 
            processing_options
        )
        
        assert result["success"] is True
        assert "ocr_results" in result
        assert "material_kai_analysis" in result
        assert result["ocr_results"]["text"] == "Extracted text from OCR"
        assert result["material_kai_analysis"]["analysis_id"] == "analysis_123"

    @pytest.mark.asyncio
    async def test_process_image_comprehensive_partial_failure(self, image_processor, sample_image_data):
        """Test comprehensive processing with partial failures."""
        # Mock OCR failure but Material Kai success
        image_processor.ocr_engine.extract_text.side_effect = Exception("OCR failed")
        
        processing_options = {
            "extract_text": True,
            "analyze_objects": True
        }
        
        result = await image_processor.process_image_comprehensive(
            sample_image_data,
            processing_options
        )
        
        assert result["success"] is True  # Partial success
        assert result["ocr_results"]["success"] is False
        assert result["material_kai_analysis"]["success"] is True
        assert "OCR failed" in result["ocr_results"]["error"]

    @pytest.mark.asyncio
    async def test_resize_image_success(self, image_processor, sample_image_data):
        """Test successful image resizing."""
        target_size = (50, 50)
        
        result = await image_processor.resize_image(sample_image_data, target_size)
        
        assert result["success"] is True
        assert "resized_data" in result
        assert result["original_size"] == (100, 100)
        assert result["new_size"] == target_size

    @pytest.mark.asyncio
    async def test_resize_image_invalid_size(self, image_processor, sample_image_data):
        """Test image resizing with invalid size."""
        invalid_size = (-10, 50)
        
        result = await image_processor.resize_image(sample_image_data, invalid_size)
        
        assert result["success"] is False
        assert "invalid" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_enhance_image_quality_success(self, image_processor, sample_image_data):
        """Test successful image quality enhancement."""
        enhancement_options = {
            "sharpen": True,
            "denoise": True,
            "brightness": 1.1,
            "contrast": 1.2
        }
        
        result = await image_processor.enhance_image_quality(
            sample_image_data,
            enhancement_options
        )
        
        assert result["success"] is True
        assert "enhanced_data" in result
        assert result["enhancements_applied"] == enhancement_options

    @pytest.mark.asyncio
    async def test_enhance_image_quality_failure(self, image_processor):
        """Test image enhancement failure."""
        corrupted_data = b"not an image"
        
        result = await image_processor.enhance_image_quality(corrupted_data)
        
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_extract_metadata_success(self, image_processor, sample_image_data):
        """Test successful metadata extraction."""
        result = await image_processor.extract_metadata(sample_image_data)
        
        assert result["success"] is True
        assert "width" in result["metadata"]
        assert "height" in result["metadata"]
        assert "format" in result["metadata"]
        assert "mode" in result["metadata"]
        assert result["metadata"]["width"] == 100
        assert result["metadata"]["height"] == 100

    @pytest.mark.asyncio
    async def test_extract_metadata_failure(self, image_processor):
        """Test metadata extraction failure."""
        corrupted_data = b"not an image"
        
        result = await image_processor.extract_metadata(corrupted_data)
        
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_batch_process_images_success(self, image_processor, sample_image_data):
        """Test successful batch image processing."""
        images = [
            {"id": "img_1", "data": sample_image_data, "filename": "test1.png"},
            {"id": "img_2", "data": sample_image_data, "filename": "test2.png"},
            {"id": "img_3", "data": sample_image_data, "filename": "test3.png"}
        ]
        
        processing_options = {
            "extract_text": True,
            "analyze_objects": False
        }
        
        result = await image_processor.batch_process_images(images, processing_options)
        
        assert result["success"] is True
        assert result["total_images"] == 3
        assert result["processed_count"] == 3
        assert result["failed_count"] == 0
        assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_batch_process_images_partial_failure(self, image_processor, sample_image_data):
        """Test batch processing with some failures."""
        images = [
            {"id": "img_1", "data": sample_image_data, "filename": "test1.png"},
            {"id": "img_2", "data": b"corrupted", "filename": "test2.png"},
            {"id": "img_3", "data": sample_image_data, "filename": "test3.png"}
        ]
        
        result = await image_processor.batch_process_images(images)
        
        assert result["success"] is True  # Partial success
        assert result["total_images"] == 3
        assert result["processed_count"] == 2
        assert result["failed_count"] == 1

    @pytest.mark.asyncio
    async def test_convert_format_success(self, image_processor, sample_image_data):
        """Test successful format conversion."""
        target_format = "JPEG"
        
        result = await image_processor.convert_format(sample_image_data, target_format)
        
        assert result["success"] is True
        assert "converted_data" in result
        assert result["original_format"] == "PNG"
        assert result["target_format"] == target_format

    @pytest.mark.asyncio
    async def test_convert_format_unsupported(self, image_processor, sample_image_data):
        """Test format conversion to unsupported format."""
        unsupported_format = "WEBP"
        
        result = await image_processor.convert_format(sample_image_data, unsupported_format)
        
        assert result["success"] is False
        assert "not supported" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_detect_faces_success(self, image_processor, sample_image_data):
        """Test successful face detection."""
        # Mock face detection via Material Kai
        image_processor.material_kai_service.analyze_image.return_value = {
            "success": True,
            "faces": [
                {
                    "bbox": [20, 20, 60, 60],
                    "confidence": 0.95,
                    "landmarks": {
                        "left_eye": [30, 35],
                        "right_eye": [50, 35],
                        "nose": [40, 45],
                        "mouth": [40, 55]
                    }
                }
            ]
        }
        
        result = await image_processor.detect_faces(sample_image_data)
        
        assert result["success"] is True
        assert len(result["faces"]) == 1
        assert result["faces"][0]["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_detect_objects_success(self, image_processor, sample_image_data):
        """Test successful object detection."""
        result = await image_processor.detect_objects(sample_image_data)
        
        assert result["success"] is True
        assert len(result["objects"]) == 2
        assert result["objects"][0]["label"] == "document"
        assert result["objects"][0]["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_search_similar_images_success(self, image_processor, sample_image_data):
        """Test successful similar image search."""
        search_options = {
            "top_k": 5,
            "similarity_threshold": 0.8
        }
        
        result = await image_processor.search_similar_images(sample_image_data, search_options)
        
        assert result["success"] is True
        assert result["search_id"] == "search_789"
        assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_generate_thumbnail_success(self, image_processor, sample_image_data):
        """Test successful thumbnail generation."""
        thumbnail_size = (32, 32)
        
        result = await image_processor.generate_thumbnail(sample_image_data, thumbnail_size)
        
        assert result["success"] is True
        assert "thumbnail_data" in result
        assert result["thumbnail_size"] == thumbnail_size

    @pytest.mark.asyncio
    async def test_analyze_color_distribution_success(self, image_processor, sample_image_data):
        """Test successful color distribution analysis."""
        result = await image_processor.analyze_color_distribution(sample_image_data)
        
        assert result["success"] is True
        assert "dominant_colors" in result
        assert "color_histogram" in result
        assert "average_color" in result

    @pytest.mark.asyncio
    async def test_extract_regions_of_interest_success(self, image_processor, sample_image_data):
        """Test successful ROI extraction."""
        result = await image_processor.extract_regions_of_interest(sample_image_data)
        
        assert result["success"] is True
        assert "regions" in result
        # Should find text regions based on Material Kai analysis
        assert len(result["regions"]) >= 1

    @pytest.mark.asyncio
    async def test_validate_image_quality_success(self, image_processor, sample_image_data):
        """Test successful image quality validation."""
        quality_criteria = {
            "min_width": 50,
            "min_height": 50,
            "max_file_size": 1024 * 1024,  # 1MB
            "allowed_formats": ["PNG", "JPEG"]
        }
        
        result = await image_processor.validate_image_quality(
            sample_image_data,
            quality_criteria
        )
        
        assert result["valid"] is True
        assert result["quality_score"] > 0

    @pytest.mark.asyncio
    async def test_validate_image_quality_failure(self, image_processor, sample_image_data):
        """Test image quality validation failure."""
        strict_criteria = {
            "min_width": 200,  # Too strict for our 100x100 test image
            "min_height": 200,
            "max_file_size": 100,  # Too small
            "allowed_formats": ["JPEG"]  # Wrong format
        }
        
        result = await image_processor.validate_image_quality(
            sample_image_data,
            strict_criteria
        )
        
        assert result["valid"] is False
        assert len(result["violations"]) > 0

    @pytest.mark.asyncio
    async def test_get_processing_statistics_success(self, image_processor):
        """Test processing statistics retrieval."""
        result = await image_processor.get_processing_statistics()
        
        assert result["success"] is True
        assert "total_processed" in result["statistics"]
        assert "success_rate" in result["statistics"]
        assert "average_processing_time" in result["statistics"]

    @pytest.mark.asyncio
    async def test_cleanup_resources_success(self, image_processor):
        """Test resource cleanup."""
        result = await image_processor.cleanup_resources()
        
        assert result["success"] is True
        assert "cleaned_up" in result["status"].lower()

    def test_configuration_validation(self, image_processor):
        """Test processor configuration validation."""
        config = image_processor.get_configuration()
        
        assert "supported_formats" in config
        assert "max_image_size" in config
        assert "quality_settings" in config
        assert isinstance(config["supported_formats"], list)

    @pytest.mark.asyncio
    async def test_error_handling_robustness(self, image_processor):
        """Test error handling with various invalid inputs."""
        # Test with None input
        result = await image_processor.validate_image_format(None, "png")
        assert result["valid"] is False
        
        # Test with empty data
        result = await image_processor.extract_text_ocr(b"")
        assert result["success"] is False
        
        # Test with invalid format
        result = await image_processor.convert_format(b"fake", "INVALID")
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_concurrent_processing_safety(self, image_processor, sample_image_data):
        """Test concurrent processing safety."""
        import asyncio
        
        # Process multiple images concurrently
        tasks = [
            image_processor.extract_text_ocr(sample_image_data)
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should succeed
        for result in results:
            assert not isinstance(result, Exception)
            assert result["success"] is True