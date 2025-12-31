"""
Image processing Pydantic schemas.

This module contains schemas for image analysis, processing,
and Material Kai Vision Platform integration.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

from .common import BaseResponse, ProcessingStatus


class ImageAnalysisRequest(BaseModel):
    """Request model for image analysis using Material Kai Vision Platform."""
    
    # Image source
    image_id: Optional[str] = Field(None, description="ID of extracted image from document")
    image_url: Optional[HttpUrl] = Field(None, description="URL to external image")
    
    # Analysis options
    analysis_types: List[str] = Field(
        default=["description", "ocr", "objects"],
        description="Types of analysis to perform"
    )
    
    # Processing parameters
    quality: str = Field("standard", pattern="^(fast|standard|high)$", description="Analysis quality")
    language: Optional[str] = Field("auto", description="Language hint for OCR")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence for results")
    
    # Context information
    document_context: Optional[str] = Field(None, description="Document context for better analysis")
    page_context: Optional[str] = Field(None, description="Page context around the image")
    
    @field_validator('analysis_types')
    @classmethod
    def validate_analysis_types(cls, v):
        valid_types = ["description", "ocr", "objects", "faces", "landmarks", "logos", "text_detection"]
        for analysis_type in v:
            if analysis_type not in valid_types:
                raise ValueError(f'Invalid analysis type: {analysis_type}. Valid types: {valid_types}')
        return v
    
    @model_validator(mode='after')
    def validate_image_source(self):
        image_id = getattr(self, 'image_id', None)
        image_url = getattr(self, 'image_url', None)
        if not image_id and not image_url:
            raise ValueError('Either image_id or image_url must be provided')
        if image_id and image_url:
            raise ValueError('Provide either image_id or image_url, not both')
        return self
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_id": "img_123e4567-e89b-12d3-a456-426614174000",
                "analysis_types": ["description", "ocr", "objects"],
                "quality": "standard",
                "language": "en",
                "confidence_threshold": 0.8,
                "document_context": "Research paper about machine learning"
            }
        }


class BoundingBox(BaseModel):
    """Bounding box coordinates for detected objects."""
    
    x: float = Field(..., description="X coordinate (normalized 0-1)")
    y: float = Field(..., description="Y coordinate (normalized 0-1)")
    width: float = Field(..., description="Width (normalized 0-1)")
    height: float = Field(..., description="Height (normalized 0-1)")
    
    @field_validator('x', 'y', 'width', 'height')
    @classmethod
    def validate_coordinates(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Coordinates must be normalized between 0 and 1')
        return v


class DetectedObject(BaseModel):
    """Detected object in image."""
    
    label: str = Field(..., description="Object label/class")
    confidence: float = Field(..., description="Detection confidence (0-1)")
    bounding_box: BoundingBox = Field(..., description="Object location")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Additional object attributes")


class DetectedText(BaseModel):
    """Detected text in image (OCR result)."""
    
    text: str = Field(..., description="Extracted text")
    confidence: float = Field(..., description="OCR confidence (0-1)")
    bounding_box: BoundingBox = Field(..., description="Text location")
    language: Optional[str] = Field(None, description="Detected language")
    font_info: Optional[Dict[str, Any]] = Field(None, description="Font characteristics")


class FaceDetection(BaseModel):
    """Detected face in image."""
    
    confidence: float = Field(..., description="Face detection confidence")
    bounding_box: BoundingBox = Field(..., description="Face location")
    landmarks: Optional[List[Dict[str, float]]] = Field(None, description="Facial landmarks")
    attributes: Optional[Dict[str, Any]] = Field(None, description="Face attributes (age, emotion, etc.)")


class ImageMetadata(BaseModel):
    """Comprehensive image metadata and analysis results."""
    
    # Basic image properties
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    format: str = Field(..., description="Image format (JPEG, PNG, etc.)")
    size_bytes: int = Field(..., description="File size in bytes")
    color_mode: Optional[str] = Field(None, description="Color mode (RGB, CMYK, etc.)")
    
    # Quality metrics
    resolution_dpi: Optional[int] = Field(None, description="Image resolution in DPI")
    quality_score: Optional[float] = Field(None, description="Overall image quality (0-1)")
    sharpness_score: Optional[float] = Field(None, description="Image sharpness (0-1)")
    
    # Content analysis
    dominant_colors: List[str] = Field(default_factory=list, description="Dominant color palette")
    brightness: Optional[float] = Field(None, description="Average brightness (0-1)")
    contrast: Optional[float] = Field(None, description="Image contrast (0-1)")
    
    # Technical metadata
    exif_data: Optional[Dict[str, Any]] = Field(None, description="EXIF metadata")
    creation_date: Optional[str] = Field(None, description="Image creation date")


class ImageAnalysisResponse(BaseResponse):
    """Response model for image analysis."""
    
    image_id: str = Field(..., description="Image identifier")
    status: ProcessingStatus = Field(..., description="Analysis status")
    
    # Analysis results
    description: Optional[str] = Field(None, description="AI-generated image description")
    detected_objects: List[DetectedObject] = Field(default_factory=list, description="Detected objects")
    detected_text: List[DetectedText] = Field(default_factory=list, description="OCR results")
    detected_faces: List[FaceDetection] = Field(default_factory=list, description="Face detection results")
    
    # Content classification
    categories: List[Dict[str, float]] = Field(default_factory=list, description="Image categories with confidence")
    tags: List[str] = Field(default_factory=list, description="Automatically generated tags")
    
    # Technical information
    metadata: ImageMetadata = Field(..., description="Image metadata and properties")
    
    # Processing information
    analysis_types_performed: List[str] = Field(..., description="Types of analysis completed")
    processing_time_ms: float = Field(..., description="Analysis processing time")
    model_versions: Dict[str, str] = Field(default_factory=dict, description="AI model versions used")
    
    # Error information (if status is failed)
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error details if analysis failed")
    
    class Config:
        model_config = {"protected_namespaces": ()}
        json_schema_extra = {
            "example": {
                "success": True,
                "image_id": "img_123e4567-e89b-12d3-a456-426614174000",
                "status": "completed",
                "description": "A diagram showing a neural network architecture with multiple layers",
                "detected_objects": [
                    {
                        "label": "diagram",
                        "confidence": 0.95,
                        "bounding_box": {
                            "x": 0.1,
                            "y": 0.2,
                            "width": 0.8,
                            "height": 0.6
                        }
                    }
                ],
                "detected_text": [
                    {
                        "text": "Neural Network",
                        "confidence": 0.98,
                        "bounding_box": {
                            "x": 0.3,
                            "y": 0.1,
                            "width": 0.4,
                            "height": 0.05
                        },
                        "language": "en"
                    }
                ],
                "categories": [
                    {"technical_diagram": 0.92},
                    {"educational_content": 0.87}
                ],
                "tags": ["neural network", "diagram", "ai", "machine learning"],
                "metadata": {
                    "width": 1024,
                    "height": 768,
                    "format": "PNG",
                    "size_bytes": 245760,
                    "quality_score": 0.89
                },
                "analysis_types_performed": ["description", "ocr", "objects"],
                "processing_time_ms": 1250.5,
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class ImageBatchRequest(BaseModel):
    """Request model for batch image analysis."""
    
    image_ids: List[str] = Field(..., min_items=1, max_items=50, description="List of image IDs to analyze")
    analysis_types: List[str] = Field(
        default=["description", "ocr"],
        description="Types of analysis to perform on all images"
    )
    
    # Batch processing options
    parallel_processing: bool = Field(True, description="Process images in parallel")
    priority: str = Field("normal", pattern="^(low|normal|high)$", description="Processing priority")
    
    # Common parameters for all images
    quality: str = Field("standard", pattern="^(fast|standard|high)$", description="Analysis quality")
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_ids": [
                    "img_123e4567-e89b-12d3-a456-426614174000",
                    "img_456e7890-e89b-12d3-a456-426614174001"
                ],
                "analysis_types": ["description", "ocr", "objects"],
                "parallel_processing": True,
                "priority": "normal",
                "quality": "standard"
            }
        }


class ImageBatchResult(BaseModel):
    """Individual result in batch processing."""
    
    image_id: str = Field(..., description="Image identifier")
    status: ProcessingStatus = Field(..., description="Processing status for this image")
    result: Optional[ImageAnalysisResponse] = Field(None, description="Analysis result if successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    processing_time_ms: float = Field(..., description="Processing time for this image")


class ImageBatchResponse(BaseResponse):
    """Response model for batch image analysis."""
    
    batch_id: str = Field(..., description="Batch processing identifier")
    total_images: int = Field(..., description="Total number of images in batch")
    
    # Processing status
    completed_count: int = Field(..., description="Number of successfully processed images")
    failed_count: int = Field(..., description="Number of failed images")
    
    # Results
    results: List[ImageBatchResult] = Field(..., description="Individual image results")
    
    # Batch metrics
    total_processing_time_ms: float = Field(..., description="Total batch processing time")
    average_time_per_image_ms: float = Field(..., description="Average processing time per image")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "batch_id": "batch_789e0123-e89b-12d3-a456-426614174000",
                "total_images": 2,
                "completed_count": 2,
                "failed_count": 0,
                "results": [
                    {
                        "image_id": "img_123",
                        "status": "completed",
                        "processing_time_ms": 1250.5
                    },
                    {
                        "image_id": "img_456",
                        "status": "completed",
                        "processing_time_ms": 980.2
                    }
                ],
                "total_processing_time_ms": 2230.7,
                "average_time_per_image_ms": 1115.35,
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class ImageSearchRequest(BaseModel):
    """Request model for image similarity search."""
    
    # Search criteria
    query_image_id: Optional[str] = Field(None, description="Find images similar to this one")
    query_description: Optional[str] = Field(None, description="Find images matching this description")
    
    # Search parameters
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    
    # Filters
    document_ids: Optional[List[str]] = Field(None, description="Limit search to specific documents")
    image_types: Optional[List[str]] = Field(None, description="Filter by image types (diagram, photo, etc.)")
    tags: Optional[List[str]] = Field(None, description="Filter by image tags")
    
    @model_validator(mode='after')
    def validate_query(self):
        query_image_id = getattr(self, 'query_image_id', None)
        query_description = getattr(self, 'query_description', None)
        if not query_image_id and not query_description:
            raise ValueError('Either query_image_id or query_description must be provided')
        return self
    
    class Config:
        json_schema_extra = {
            "example": {
                "query_description": "neural network diagram",
                "limit": 15,
                "similarity_threshold": 0.8,
                "image_types": ["diagram", "chart"],
                "tags": ["ai", "machine learning"]
            }
        }


class SimilarImage(BaseModel):
    """Similar image search result."""
    
    image_id: str = Field(..., description="Image identifier")
    document_id: str = Field(..., description="Source document ID")
    document_name: str = Field(..., description="Source document name")
    page_number: int = Field(..., description="Page number where image appears")
    
    # Similarity metrics
    similarity_score: float = Field(..., description="Visual similarity score (0-1)")
    content_similarity: Optional[float] = Field(None, description="Content/description similarity")
    
    # Image information
    description: Optional[str] = Field(None, description="Image description")
    tags: List[str] = Field(default_factory=list, description="Image tags")
    dimensions: Dict[str, int] = Field(..., description="Image dimensions")
    
    # Access information
    image_url: Optional[str] = Field(None, description="URL to access the image")


class ImageSearchResponse(BaseResponse):
    """Response model for image search."""
    
    query_info: Dict[str, Any] = Field(..., description="Information about the search query")
    similar_images: List[SimilarImage] = Field(..., description="Similar images found")
    total_found: int = Field(..., description="Total number of similar images")
    search_time_ms: float = Field(..., description="Search execution time")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "query_info": {
                    "type": "description",
                    "query": "neural network diagram"
                },
                "similar_images": [
                    {
                        "image_id": "img_789",
                        "document_id": "doc_123",
                        "document_name": "Deep Learning Paper",
                        "page_number": 5,
                        "similarity_score": 0.92,
                        "description": "Convolutional neural network architecture diagram",
                        "tags": ["cnn", "architecture", "diagram"],
                        "dimensions": {"width": 800, "height": 600}
                    }
                ],
                "total_found": 8,
                "search_time_ms": 245.7,
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }
