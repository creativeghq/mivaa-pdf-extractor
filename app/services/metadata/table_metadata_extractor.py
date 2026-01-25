"""
Table Metadata Extractor Service

Parses extracted tables from product_tables to enrich product metadata with:
- Dimensions (width, height, thickness, sizes)
- Packaging info (pieces per box, boxes per pallet, weight, coverage)
- Performance specs (slip resistance, water absorption, etc.)
- Loading info (pallet weight, truck capacity)
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ParsedDimension:
    """Parsed dimension with unit"""
    width: Optional[float] = None
    height: Optional[float] = None
    thickness: Optional[float] = None
    unit: str = "cm"
    format_string: Optional[str] = None  # e.g., "60x60 cm"

    def to_dict(self) -> Dict[str, Any]:
        result = {"unit": self.unit}
        if self.width:
            result["width"] = self.width
        if self.height:
            result["height"] = self.height
        if self.thickness:
            result["thickness"] = self.thickness
        if self.format_string:
            result["format"] = self.format_string
        return result


class TableMetadataExtractor:
    """
    Extracts structured metadata from product tables.

    Handles multiple table types:
    - dimensions: tile sizes, thickness
    - packaging: boxes, pallets, weight, coverage
    - specifications: performance metrics
    - pricing: costs (not extracted for security)
    """

    def __init__(self, supabase_client: Any):
        self.supabase = supabase_client
        self.logger = logger

    async def extract_metadata_from_tables(
        self,
        product_id: str
    ) -> Dict[str, Any]:
        """
        Extract all metadata from tables associated with a product.

        Args:
            product_id: Product UUID

        Returns:
            Dictionary with extracted metadata:
            {
                "dimensions": [...],
                "packaging": {...},
                "performance": {...},
                "available_sizes": [...],
                "thickness": "...",
                ...
            }
        """
        try:
            # Fetch all tables for this product
            tables_response = self.supabase.client.table('product_tables')\
                .select('*')\
                .eq('product_id', product_id)\
                .execute()

            if not tables_response.data:
                self.logger.debug(f"No tables found for product {product_id}")
                return {}

            tables = tables_response.data
            self.logger.info(f"Processing {len(tables)} tables for product {product_id}")

            # Initialize result containers
            result = {
                "dimensions": [],
                "available_sizes": [],
                "packaging": {},
                "performance": {},
                "specifications": {},
                "_table_sources": []  # Track which tables contributed data
            }

            # Process each table by type
            for table in tables:
                table_type = table.get('table_type', 'other')
                table_data = table.get('table_data', {})
                headers = table.get('headers', [])
                page_num = table.get('page_number', 0)

                self.logger.debug(f"   Processing {table_type} table from page {page_num}")

                if table_type == 'dimensions':
                    dims = self._parse_dimensions_table(headers, table_data)
                    if dims:
                        result["dimensions"].extend(dims)
                        result["_table_sources"].append({
                            "page": page_num,
                            "type": "dimensions",
                            "extracted": len(dims)
                        })

                elif table_type == 'specifications':
                    specs = self._parse_specifications_table(headers, table_data)
                    result["specifications"].update(specs)
                    result["performance"].update(specs.get("performance", {}))
                    result["_table_sources"].append({
                        "page": page_num,
                        "type": "specifications"
                    })

                elif table_type == 'packaging' or self._looks_like_packaging(headers, table_data):
                    pkg = self._parse_packaging_table(headers, table_data)
                    if pkg:
                        # Merge packaging data (may have multiple packaging tables)
                        for key, value in pkg.items():
                            if key not in result["packaging"] or not result["packaging"][key]:
                                result["packaging"][key] = value
                        result["_table_sources"].append({
                            "page": page_num,
                            "type": "packaging"
                        })

                else:
                    # Try to extract any dimensions or specs from unknown tables
                    generic = self._parse_generic_table(headers, table_data)
                    if generic.get("dimensions"):
                        result["dimensions"].extend(generic["dimensions"])
                    if generic.get("packaging"):
                        result["packaging"].update(generic["packaging"])

            # Post-process: Convert dimensions to available_sizes format
            if result["dimensions"]:
                result["available_sizes"] = self._dimensions_to_sizes(result["dimensions"])

                # Extract thickness if found
                thickness_values = [d.get("thickness") for d in result["dimensions"] if d.get("thickness")]
                if thickness_values:
                    # Use most common thickness
                    result["thickness"] = f"{max(set(thickness_values), key=thickness_values.count)} mm"

            self.logger.info(f"   Extracted from tables: {len(result['available_sizes'])} sizes, "
                           f"{len(result['packaging'])} packaging fields, "
                           f"{len(result['performance'])} performance specs")

            return result

        except Exception as e:
            self.logger.error(f"Table metadata extraction failed: {e}")
            return {}

    def _parse_dimensions_table(
        self,
        headers: List[str],
        table_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Parse a dimensions table into structured dimension data."""
        dimensions = []

        # Normalize headers
        headers_lower = [h.lower() if h else "" for h in headers]

        # Find relevant columns
        size_col = self._find_column(headers_lower, ['size', 'dimension', 'format', 'formato'])
        width_col = self._find_column(headers_lower, ['width', 'w', 'ancho', 'largo'])
        height_col = self._find_column(headers_lower, ['height', 'h', 'length', 'l', 'alto'])
        thickness_col = self._find_column(headers_lower, ['thickness', 'th', 'espesor', 'spessore'])

        # Get rows from table_data
        rows = table_data.get('rows', []) or table_data.get('data', [])
        if not rows and isinstance(table_data, list):
            rows = table_data

        for row in rows:
            if not row or not isinstance(row, (list, dict)):
                continue

            # Handle list rows
            if isinstance(row, list):
                dim = {}

                # Try to extract from size column (e.g., "60x60 cm")
                if size_col is not None and size_col < len(row):
                    size_str = str(row[size_col])
                    parsed = self._parse_size_string(size_str)
                    if parsed:
                        dim.update(parsed)

                # Try explicit width/height columns
                if width_col is not None and width_col < len(row):
                    dim["width"] = self._extract_number(row[width_col])
                if height_col is not None and height_col < len(row):
                    dim["height"] = self._extract_number(row[height_col])
                if thickness_col is not None and thickness_col < len(row):
                    dim["thickness"] = self._extract_number(row[thickness_col])

                if dim.get("width") or dim.get("height"):
                    dim.setdefault("unit", "cm")
                    dimensions.append(dim)

            # Handle dict rows
            elif isinstance(row, dict):
                dim = {}
                for key, value in row.items():
                    key_lower = key.lower()
                    if any(s in key_lower for s in ['size', 'dimension', 'format']):
                        parsed = self._parse_size_string(str(value))
                        if parsed:
                            dim.update(parsed)
                    elif any(s in key_lower for s in ['width', 'ancho']):
                        dim["width"] = self._extract_number(value)
                    elif any(s in key_lower for s in ['height', 'length', 'alto']):
                        dim["height"] = self._extract_number(value)
                    elif any(s in key_lower for s in ['thickness', 'espesor']):
                        dim["thickness"] = self._extract_number(value)

                if dim.get("width") or dim.get("height"):
                    dim.setdefault("unit", "cm")
                    dimensions.append(dim)

        return dimensions

    def _parse_packaging_table(
        self,
        headers: List[str],
        table_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse a packaging table into structured data."""
        packaging = {}

        # Normalize headers
        headers_lower = [h.lower() if h else "" for h in headers]

        # Define packaging field mappings
        field_mappings = {
            "pieces_per_box": ['pieces', 'pcs', 'pezzi', 'piezas', 'box', 'caja'],
            "boxes_per_pallet": ['boxes', 'cartons', 'cajas', 'pallet'],
            "weight_per_box_kg": ['weight', 'peso', 'kg'],
            "coverage_per_box_m2": ['coverage', 'm2', 'm2', 'sqm', 'area'],
            "pallet_weight_kg": ['pallet weight', 'peso pallet'],
            "pieces_per_m2": ['pieces/m2', 'pcs/m2', 'piezas/m2']
        }

        # Get rows from table_data
        rows = table_data.get('rows', []) or table_data.get('data', [])
        if not rows and isinstance(table_data, list):
            rows = table_data

        # Try to match columns to fields
        for field, keywords in field_mappings.items():
            col_idx = self._find_column(headers_lower, keywords)
            if col_idx is not None:
                # Get value from first data row
                for row in rows:
                    if isinstance(row, list) and col_idx < len(row):
                        value = self._extract_number(row[col_idx])
                        if value:
                            packaging[field] = value
                            break
                    elif isinstance(row, dict):
                        for key, val in row.items():
                            if any(kw in key.lower() for kw in keywords):
                                packaging[field] = self._extract_number(val)
                                break

        return packaging

    def _parse_specifications_table(
        self,
        headers: List[str],
        table_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse a specifications table into structured data."""
        specs = {"performance": {}}

        # Get rows
        rows = table_data.get('rows', []) or table_data.get('data', [])
        if not rows and isinstance(table_data, list):
            rows = table_data

        # Spec field mappings
        spec_mappings = {
            "slip_resistance": ['slip', 'r10', 'r11', 'r12', 'anti-slip', 'antideslizante'],
            "water_absorption": ['water absorption', 'absorcion', 'assorbimento'],
            "breaking_strength": ['breaking', 'ruptura', 'rottura'],
            "frost_resistance": ['frost', 'helada', 'gelo'],
            "abrasion_resistance": ['abrasion', 'pei', 'abrasion'],
            "chemical_resistance": ['chemical', 'quimico', 'chimico'],
            "fire_rating": ['fire', 'fuego', 'fuoco']
        }

        for row in rows:
            if isinstance(row, dict):
                for key, value in row.items():
                    key_lower = key.lower()
                    for spec_field, keywords in spec_mappings.items():
                        if any(kw in key_lower for kw in keywords):
                            specs["performance"][spec_field] = str(value)
                            break
            elif isinstance(row, list) and len(row) >= 2:
                # Assume property-value format
                prop = str(row[0]).lower()
                value = str(row[1])
                for spec_field, keywords in spec_mappings.items():
                    if any(kw in prop for kw in keywords):
                        specs["performance"][spec_field] = value
                        break

        return specs

    def _parse_generic_table(
        self,
        headers: List[str],
        table_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Try to extract any useful data from an unclassified table."""
        result = {"dimensions": [], "packaging": {}}

        # Get all text from table
        rows = table_data.get('rows', []) or table_data.get('data', [])
        if not rows and isinstance(table_data, list):
            rows = table_data

        for row in rows:
            row_text = ""
            if isinstance(row, list):
                row_text = " ".join(str(cell) for cell in row if cell)
            elif isinstance(row, dict):
                row_text = " ".join(str(v) for v in row.values() if v)

            # Look for dimension patterns
            dim_match = re.search(r'(\d+(?:[.,]\d+)?)\s*[x]\s*(\d+(?:[.,]\d+)?)\s*(cm|mm)?', row_text, re.IGNORECASE)
            if dim_match:
                width = float(dim_match.group(1).replace(',', '.'))
                height = float(dim_match.group(2).replace(',', '.'))
                unit = dim_match.group(3) or 'cm'
                result["dimensions"].append({
                    "width": width,
                    "height": height,
                    "unit": unit,
                    "format": f"{width}x{height} {unit}"
                })

            # Look for packaging patterns
            pieces_match = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:pcs|pieces|piezas|pezzi)/(?:box|caja)', row_text, re.IGNORECASE)
            if pieces_match:
                result["packaging"]["pieces_per_box"] = float(pieces_match.group(1).replace(',', '.'))

            weight_match = re.search(r'(\d+(?:[.,]\d+)?)\s*kg/?(?:box|caja)?', row_text, re.IGNORECASE)
            if weight_match:
                result["packaging"]["weight_per_box_kg"] = float(weight_match.group(1).replace(',', '.'))

        return result

    def _looks_like_packaging(self, headers: List[str], table_data: Dict[str, Any]) -> bool:
        """Check if a table looks like packaging info based on content."""
        headers_text = " ".join(h.lower() for h in headers if h)
        packaging_keywords = ['box', 'pallet', 'weight', 'coverage', 'pcs', 'pieces', 'kg', 'm2']
        return any(kw in headers_text for kw in packaging_keywords)

    def _find_column(self, headers: List[str], keywords: List[str]) -> Optional[int]:
        """Find column index matching any of the keywords."""
        for idx, header in enumerate(headers):
            if any(kw in header for kw in keywords):
                return idx
        return None

    def _parse_size_string(self, size_str: str) -> Optional[Dict[str, Any]]:
        """Parse a size string like '60x60 cm' or '30x60x0.8cm'."""
        if not size_str:
            return None

        # Pattern for WxH or WxHxT with optional unit
        pattern = r'(\d+(?:[.,]\d+)?)\s*[x]\s*(\d+(?:[.,]\d+)?)\s*(?:[x]\s*(\d+(?:[.,]\d+)?))?\s*(cm|mm)?'
        match = re.search(pattern, size_str, re.IGNORECASE)

        if match:
            width = float(match.group(1).replace(',', '.'))
            height = float(match.group(2).replace(',', '.'))
            thickness = float(match.group(3).replace(',', '.')) if match.group(3) else None
            unit = match.group(4) or 'cm'

            result = {
                "width": width,
                "height": height,
                "unit": unit,
                "format": f"{int(width) if width == int(width) else width}x{int(height) if height == int(height) else height} {unit}"
            }
            if thickness:
                result["thickness"] = thickness
            return result

        return None

    def _extract_number(self, value: Any) -> Optional[float]:
        """Extract a numeric value from various formats."""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)

        # Try to extract number from string
        str_val = str(value).replace(',', '.')
        match = re.search(r'(\d+(?:\.\d+)?)', str_val)
        if match:
            return float(match.group(1))
        return None

    def _dimensions_to_sizes(self, dimensions: List[Dict[str, Any]]) -> List[str]:
        """Convert dimension objects to size strings."""
        sizes = set()
        for dim in dimensions:
            if dim.get("format"):
                sizes.add(dim["format"])
            elif dim.get("width") and dim.get("height"):
                w = dim["width"]
                h = dim["height"]
                unit = dim.get("unit", "cm")
                sizes.add(f"{int(w) if w == int(w) else w}x{int(h) if h == int(h) else h} {unit}")
        return sorted(list(sizes))


async def enrich_product_from_tables(
    product_id: str,
    supabase_client: Any
) -> Dict[str, Any]:
    """
    Convenience function to extract table metadata and merge with existing product metadata.

    Args:
        product_id: Product UUID
        supabase_client: Supabase client

    Returns:
        Merged metadata dictionary
    """
    extractor = TableMetadataExtractor(supabase_client)
    table_metadata = await extractor.extract_metadata_from_tables(product_id)
    return table_metadata
