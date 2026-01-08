"""
Table Extraction Service using Camelot

This service extracts tables from PDFs using Camelot library, guided by YOLO layout regions.
Tables are extracted as structured JSON and stored in the product_tables database table.

Features:
- Uses YOLO TABLE region bounding boxes to guide extraction
- Supports both lattice (bordered) and stream (borderless) table detection
- Converts tables to structured JSON format
- Stores in product_tables table with metadata
"""

import logging
from typing import List, Dict, Any, Optional
import camelot
from pathlib import Path

logger = logging.getLogger(__name__)


class TableExtractor:
    """
    Extract tables from PDFs using Camelot, guided by YOLO layout regions.
    """
    
    def __init__(self):
        """Initialize table extractor."""
        self.logger = logger
    
    def extract_tables_from_page(
        self,
        pdf_path: str,
        page_number: int,
        table_regions: Optional[List[Dict[str, Any]]] = None,
        flavor: str = 'lattice'
    ) -> List[Dict[str, Any]]:
        """
        Extract tables from a specific PDF page.
        
        Args:
            pdf_path: Path to PDF file
            page_number: Page number (1-based)
            table_regions: Optional YOLO TABLE regions with bounding boxes
            flavor: 'lattice' for bordered tables, 'stream' for borderless
            
        Returns:
            List of extracted tables as dictionaries
        """
        try:
            # Validate PDF exists
            if not Path(pdf_path).exists():
                raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
            extracted_tables = []
            
            # If YOLO regions provided, use them to guide extraction
            if table_regions and len(table_regions) > 0:
                self.logger.info(f"   📊 Extracting {len(table_regions)} tables from page {page_number} using YOLO regions")
                
                for region in table_regions:
                    # Extract bounding box coordinates
                    bbox = region.get('bbox', {})
                    x1 = bbox.get('x1', 0)
                    y1 = bbox.get('y1', 0)
                    x2 = bbox.get('x2', 0)
                    y2 = bbox.get('y2', 0)
                    
                    # Camelot uses table_areas parameter: "x1,y1,x2,y2" format
                    # Note: Camelot coordinates are in points from bottom-left
                    table_area = f"{x1},{y1},{x2},{y2}"
                    
                    try:
                        # Extract table using Camelot with region guidance
                        tables = camelot.read_pdf(
                            pdf_path,
                            pages=str(page_number),
                            flavor=flavor,
                            table_areas=[table_area],
                            suppress_stdout=True
                        )
                        
                        # Convert each table to structured format
                        for table in tables:
                            table_dict = self._convert_table_to_dict(
                                table=table,
                                page_number=page_number,
                                layout_region_id=region.get('id'),
                                confidence=region.get('confidence', 0.0)
                            )
                            extracted_tables.append(table_dict)
                            
                    except Exception as e:
                        self.logger.warning(f"   ⚠️ Failed to extract table from region {region.get('id')}: {e}")
                        continue
            
            else:
                # No YOLO regions - extract all tables from page
                self.logger.info(f"   📊 Extracting tables from page {page_number} (no YOLO guidance)")
                
                try:
                    tables = camelot.read_pdf(
                        pdf_path,
                        pages=str(page_number),
                        flavor=flavor,
                        suppress_stdout=True
                    )
                    
                    for table in tables:
                        table_dict = self._convert_table_to_dict(
                            table=table,
                            page_number=page_number
                        )
                        extracted_tables.append(table_dict)
                        
                except Exception as e:
                    self.logger.warning(f"   ⚠️ Failed to extract tables from page {page_number}: {e}")
            
            self.logger.info(f"   ✅ Extracted {len(extracted_tables)} tables from page {page_number}")
            return extracted_tables
            
        except Exception as e:
            self.logger.error(f"❌ Table extraction failed for page {page_number}: {e}")
            return []
    
    def _convert_table_to_dict(
        self,
        table: Any,
        page_number: int,
        layout_region_id: Optional[str] = None,
        confidence: float = 0.0
    ) -> Dict[str, Any]:
        """
        Convert Camelot table to structured dictionary format.
        
        Args:
            table: Camelot table object
            page_number: Page number
            layout_region_id: Optional YOLO region ID
            confidence: YOLO detection confidence
            
        Returns:
            Structured table dictionary
        """
        # Convert table to pandas DataFrame
        df = table.df
        
        # Extract headers (first row)
        headers = df.iloc[0].tolist() if len(df) > 0 else []
        
        # Extract data rows (skip header row)
        data_rows = df.iloc[1:].values.tolist() if len(df) > 1 else []
        
        # Create structured table data
        table_data = {
            'headers': headers,
            'rows': data_rows,
            'num_rows': len(data_rows),
            'num_cols': len(headers)
        }
        
        return {
            'page_number': page_number,
            'layout_region_id': layout_region_id,
            'table_data': table_data,
            'headers': headers,
            'confidence': confidence,
            'extractor': 'camelot',
            'metadata': {
                'accuracy': table.accuracy if hasattr(table, 'accuracy') else None,
                'whitespace': table.whitespace if hasattr(table, 'whitespace') else None,
                'parsing_report': table.parsing_report if hasattr(table, 'parsing_report') else None
            }
        }

    def classify_table_type(self, headers: List[str], table_data: Dict[str, Any]) -> str:
        """
        Classify table type based on headers and content.

        Args:
            headers: Table headers
            table_data: Table data dictionary

        Returns:
            Table type: 'specifications', 'pricing', 'comparison', 'dimensions', 'other'
        """
        # Convert headers to lowercase for matching
        headers_lower = [str(h).lower() for h in headers]
        headers_text = ' '.join(headers_lower)

        # Specifications table keywords
        if any(keyword in headers_text for keyword in ['specification', 'property', 'feature', 'characteristic', 'parameter']):
            return 'specifications'

        # Pricing table keywords
        if any(keyword in headers_text for keyword in ['price', 'cost', 'rate', 'pricing', 'quote', 'msrp']):
            return 'pricing'

        # Comparison table keywords
        if any(keyword in headers_text for keyword in ['comparison', 'vs', 'versus', 'compare', 'model']):
            return 'comparison'

        # Dimensions table keywords
        if any(keyword in headers_text for keyword in ['dimension', 'size', 'width', 'height', 'length', 'diameter', 'thickness']):
            return 'dimensions'

        # Default to 'other'
        return 'other'

    async def store_tables_in_database(
        self,
        product_id: str,
        tables: List[Dict[str, Any]],
        supabase_client: Any
    ) -> int:
        """
        Store extracted tables in product_tables database table.

        Args:
            product_id: Product ID
            tables: List of extracted table dictionaries
            supabase_client: Supabase client instance

        Returns:
            Number of tables stored
        """
        if not tables or len(tables) == 0:
            return 0

        try:
            stored_count = 0

            for table in tables:
                # Classify table type
                table_type = self.classify_table_type(
                    headers=table.get('headers', []),
                    table_data=table.get('table_data', {})
                )

                # Prepare record for database
                table_record = {
                    'product_id': product_id,
                    'page_number': table['page_number'],
                    'layout_region_id': table.get('layout_region_id'),
                    'table_data': table['table_data'],
                    'headers': table['headers'],
                    'table_type': table_type,
                    'confidence': table.get('confidence', 0.0),
                    'extractor': table.get('extractor', 'camelot'),
                    'metadata': table.get('metadata', {})
                }

                # Insert into database
                result = supabase_client.client.table('product_tables')\
                    .insert(table_record)\
                    .execute()

                if result.data:
                    stored_count += 1
                    self.logger.debug(f"   ✅ Stored {table_type} table from page {table['page_number']}")

            self.logger.info(f"✅ Stored {stored_count} tables in database for product {product_id}")
            return stored_count

        except Exception as e:
            self.logger.error(f"❌ Failed to store tables in database: {e}")
            return 0


