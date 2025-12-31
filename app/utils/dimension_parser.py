"""
Dimension Parser Utility

Extracts and parses product dimensions from text content.
Handles various formats: 15×38, 20x40, 11.8×11.8 cm, etc.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class Dimension:
    """Structured dimension data"""
    width: float
    height: Optional[float] = None
    depth: Optional[float] = None
    unit: str = "cm"
    raw_text: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "width": self.width,
            "unit": self.unit,
            "raw_text": self.raw_text
        }
        if self.height is not None:
            result["height"] = self.height
        if self.depth is not None:
            result["depth"] = self.depth
        return result
    
    def to_string(self) -> str:
        """Convert to readable string"""
        if self.depth:
            return f"{self.width}×{self.height}×{self.depth} {self.unit}"
        elif self.height:
            return f"{self.width}×{self.height} {self.unit}"
        else:
            return f"{self.width} {self.unit}"


class DimensionParser:
    """Parse dimensions from text content"""
    
    # Dimension patterns (ordered by specificity)
    PATTERNS = [
        # 3D dimensions: 15×38×2.5 cm, 20x40x3 mm
        r'(\d+(?:\.\d+)?)\s*[×x]\s*(\d+(?:\.\d+)?)\s*[×x]\s*(\d+(?:\.\d+)?)\s*(cm|mm|m|in|inch)?',
        # 2D dimensions: 15×38 cm, 20x40 mm, 11.8×11.8
        r'(\d+(?:\.\d+)?)\s*[×x]\s*(\d+(?:\.\d+)?)\s*(cm|mm|m|in|inch)?',
        # Single dimension with unit: 15 cm, 20mm
        r'(\d+(?:\.\d+)?)\s*(cm|mm|m|in|inch)',
    ]
    
    @staticmethod
    def parse_dimension(text: str) -> Optional[Dimension]:
        """
        Parse a single dimension from text.
        
        Args:
            text: Text containing dimension (e.g., "15×38 cm")
            
        Returns:
            Dimension object or None if no dimension found
        """
        text = text.strip()
        
        for pattern in DimensionParser.PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                
                # Extract unit (default to cm)
                unit = groups[-1] if groups[-1] else "cm"
                unit = unit.lower()
                
                # Parse dimensions based on number of numeric groups
                numeric_groups = [g for g in groups[:-1] if g and g.replace('.', '').isdigit()]
                
                if len(numeric_groups) == 3:
                    # 3D dimension
                    return Dimension(
                        width=float(numeric_groups[0]),
                        height=float(numeric_groups[1]),
                        depth=float(numeric_groups[2]),
                        unit=unit,
                        raw_text=match.group(0)
                    )
                elif len(numeric_groups) == 2:
                    # 2D dimension
                    return Dimension(
                        width=float(numeric_groups[0]),
                        height=float(numeric_groups[1]),
                        unit=unit,
                        raw_text=match.group(0)
                    )
                elif len(numeric_groups) == 1:
                    # 1D dimension
                    return Dimension(
                        width=float(numeric_groups[0]),
                        unit=unit,
                        raw_text=match.group(0)
                    )
        
        return None
    
    @staticmethod
    def extract_all_dimensions(text: str) -> List[Dimension]:
        """
        Extract all dimensions from text.
        
        Args:
            text: Text content to search
            
        Returns:
            List of Dimension objects
        """
        dimensions = []
        
        for pattern in DimensionParser.PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                
                # Extract unit (default to cm)
                unit = groups[-1] if groups[-1] else "cm"
                unit = unit.lower() if unit else "cm"
                
                # Parse dimensions
                numeric_groups = [g for g in groups[:-1] if g and g.replace('.', '').isdigit()]
                
                if len(numeric_groups) >= 2:
                    dim = Dimension(
                        width=float(numeric_groups[0]),
                        height=float(numeric_groups[1]),
                        depth=float(numeric_groups[2]) if len(numeric_groups) == 3 else None,
                        unit=unit,
                        raw_text=match.group(0)
                    )
                    dimensions.append(dim)
        
        return dimensions
    
    @staticmethod
    def deduplicate_dimensions(dimensions: List[Dimension]) -> List[Dimension]:
        """
        Remove duplicate dimensions.
        
        Args:
            dimensions: List of Dimension objects
            
        Returns:
            Deduplicated list
        """
        seen = set()
        unique = []
        
        for dim in dimensions:
            # Create unique key
            key = (dim.width, dim.height, dim.depth, dim.unit)
            if key not in seen:
                seen.add(key)
                unique.append(dim)
        
        return unique


