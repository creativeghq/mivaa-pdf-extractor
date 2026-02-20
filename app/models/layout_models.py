"""
Layout Detection Models

Pydantic models for YOLO DocParser layout detection results.
Supports detection of TEXT, IMAGE, TABLE, TITLE, and CAPTION regions.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime
from uuid import UUID


class BoundingBox(BaseModel):
    """Bounding box coordinates for a detected region."""
    
    x: float = Field(..., description="X coordinate (left edge)", ge=0.0)
    y: float = Field(..., description="Y coordinate (top edge)", ge=0.0)
    width: float = Field(..., description="Width of the region", gt=0.0)
    height: float = Field(..., description="Height of the region", gt=0.0)
    page: int = Field(..., description="Page number (0-based)", ge=0)
    
    @property
    def x2(self) -> float:
        """Right edge X coordinate."""
        return self.x + self.width
    
    @property
    def y2(self) -> float:
        """Bottom edge Y coordinate."""
        return self.y + self.height
    
    @property
    def center_x(self) -> float:
        """Center X coordinate."""
        return self.x + (self.width / 2)
    
    @property
    def center_y(self) -> float:
        """Center Y coordinate."""
        return self.y + (self.height / 2)
    
    @property
    def area(self) -> float:
        """Area of the bounding box."""
        return self.width * self.height
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary format."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "page": self.page
        }


RegionType = Literal["TEXT", "IMAGE", "TABLE", "TITLE", "CAPTION"]


class LayoutRegion(BaseModel):
    """Detected layout region from YOLO DocParser."""
    
    type: RegionType = Field(..., description="Region type: TEXT, IMAGE, TABLE, TITLE, or CAPTION")
    bbox: BoundingBox = Field(..., description="Bounding box coordinates")
    confidence: float = Field(..., description="Detection confidence score", ge=0.0, le=1.0)
    text_content: Optional[str] = Field(None, description="Extracted text content (for TEXT/TITLE/CAPTION regions)")
    reading_order: Optional[int] = Field(None, description="Reading order index (top-to-bottom, left-to-right)")
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('type')
    @classmethod
    def validate_type(cls, v):
        """Validate region type."""
        allowed = ['TEXT', 'IMAGE', 'TABLE', 'TITLE', 'CAPTION']
        if v not in allowed:
            raise ValueError(f"Type must be one of {allowed}")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for database storage."""
        return {
            "type": self.type,
            "bbox_x": self.bbox.x,
            "bbox_y": self.bbox.y,
            "bbox_width": self.bbox.width,
            "bbox_height": self.bbox.height,
            "confidence": self.confidence,
            "text_content": self.text_content,
            "reading_order": self.reading_order,
            "metadata": self.metadata
        }


class LayoutDetectionResult(BaseModel):
    """Complete layout detection result for a page."""
    
    page_number: int = Field(..., description="Page number (0-based)", ge=0)
    regions: List[LayoutRegion] = Field(default_factory=list, description="Detected layout regions")
    
    # Statistics
    total_regions: int = Field(0, description="Total number of detected regions")
    text_regions: int = Field(0, description="Number of TEXT regions")
    image_regions: int = Field(0, description="Number of IMAGE regions")
    table_regions: int = Field(0, description="Number of TABLE regions")
    title_regions: int = Field(0, description="Number of TITLE regions")
    caption_regions: int = Field(0, description="Number of CAPTION regions")
    
    # Processing metadata
    detection_time_ms: Optional[int] = Field(None, description="Detection time in milliseconds")
    model_version: Optional[str] = Field(None, description="YOLO model version used")
    
    def __init__(self, **data):
        """Initialize and calculate statistics."""
        super().__init__(**data)
        self._calculate_statistics()
    
    def _calculate_statistics(self):
        """Calculate region statistics."""
        self.total_regions = len(self.regions)
        self.text_regions = sum(1 for r in self.regions if r.type == "TEXT")
        self.image_regions = sum(1 for r in self.regions if r.type == "IMAGE")
        self.table_regions = sum(1 for r in self.regions if r.type == "TABLE")
        self.title_regions = sum(1 for r in self.regions if r.type == "TITLE")
        self.caption_regions = sum(1 for r in self.regions if r.type == "CAPTION")
    
    def get_regions_by_type(self, region_type: RegionType) -> List[LayoutRegion]:
        """Get all regions of a specific type."""
        return [r for r in self.regions if r.type == region_type]
    
    def get_regions_sorted_by_reading_order(self) -> List[LayoutRegion]:
        """Get regions sorted by reading order."""
        return sorted(
            [r for r in self.regions if r.reading_order is not None],
            key=lambda r: r.reading_order
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "page_number": self.page_number,
            "regions": [r.to_dict() for r in self.regions],
            "total_regions": self.total_regions,
            "text_regions": self.text_regions,
            "image_regions": self.image_regions,
            "table_regions": self.table_regions,
            "title_regions": self.title_regions,
            "caption_regions": self.caption_regions,
            "detection_time_ms": self.detection_time_ms,
            "model_version": self.model_version
        }

