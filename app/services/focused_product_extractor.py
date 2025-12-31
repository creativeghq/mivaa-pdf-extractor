"""
Focused Product Extractor

Extracts and processes ONLY specific product pages from a PDF catalog,
rather than processing the entire document.

This is used for focused testing and validation of the AI pipeline
on a single product.
"""

import logging
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
import re

logger = logging.getLogger(__name__)


class FocusedProductExtractor:
    """Extract specific product pages from PDF catalogs."""
    
    def __init__(self):
        self.logger = logger
    
    def find_product_pages(
        self,
        pdf_path: str,
        product_name: str,
        designer: Optional[str] = None,
        search_terms: Optional[List[str]] = None
    ) -> List[int]:
        """
        Find all pages containing a specific product.
        
        Args:
            pdf_path: Path to PDF file
            product_name: Product name to search for (e.g., "NOVA")
            designer: Optional designer name (e.g., "SG NY")
            search_terms: Additional search terms
            
        Returns:
            List of page numbers (0-based) containing the product
        """
        try:
            doc = fitz.open(pdf_path)
            matching_pages = []
            
            # Build search patterns
            patterns = [product_name.lower()]
            if designer:
                patterns.append(designer.lower())
            if search_terms:
                patterns.extend([term.lower() for term in search_terms])
            
            self.logger.info(f"🔍 Searching for product '{product_name}' in {doc.page_count} pages")
            self.logger.info(f"   Search patterns: {patterns}")
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text = page.get_text().lower()
                
                # Check if any pattern matches
                matches = []
                for pattern in patterns:
                    if pattern in text:
                        matches.append(pattern)
                
                if matches:
                    matching_pages.append(page_num)
                    self.logger.info(f"   ✅ Page {page_num + 1}: Found {matches}")
            
            doc.close()
            
            self.logger.info(f"✅ Found {len(matching_pages)} pages containing '{product_name}'")
            return matching_pages
            
        except Exception as e:
            self.logger.error(f"Failed to find product pages: {e}", exc_info=True)
            raise
    
    def extract_product_pages_to_pdf(
        self,
        source_pdf_path: str,
        output_pdf_path: str,
        page_numbers: List[int]
    ) -> str:
        """
        Extract specific pages to a new PDF file.
        
        Args:
            source_pdf_path: Source PDF path
            output_pdf_path: Output PDF path
            page_numbers: List of page numbers to extract (0-based)
            
        Returns:
            Path to the new PDF file
        """
        try:
            source_doc = fitz.open(source_pdf_path)
            output_doc = fitz.open()  # Create new PDF
            
            self.logger.info(f"📄 Extracting {len(page_numbers)} pages to new PDF")
            
            for page_num in sorted(page_numbers):
                if 0 <= page_num < source_doc.page_count:
                    output_doc.insert_pdf(
                        source_doc,
                        from_page=page_num,
                        to_page=page_num
                    )
                    self.logger.info(f"   ✅ Extracted page {page_num + 1}")
            
            output_doc.save(output_pdf_path)
            output_doc.close()
            source_doc.close()
            
            self.logger.info(f"✅ Created focused PDF: {output_pdf_path}")
            return output_pdf_path
            
        except Exception as e:
            self.logger.error(f"Failed to extract pages: {e}", exc_info=True)
            raise
    
    def get_product_metadata_from_pages(
        self,
        pdf_path: str,
        page_numbers: List[int],
        product_name: str
    ) -> Dict[str, Any]:
        """
        Extract product metadata from the identified pages.
        
        Args:
            pdf_path: Path to PDF file
            page_numbers: Pages containing the product
            product_name: Product name
            
        Returns:
            Dictionary with product metadata
        """
        try:
            doc = fitz.open(pdf_path)
            
            metadata = {
                "product_name": product_name,
                "page_numbers": [p + 1 for p in page_numbers],  # Convert to 1-based
                "total_pages": len(page_numbers),
                "extracted_text": [],
                "dimensions": [],
                "materials": [],
                "colors": [],
                "designers": []
            }
            
            # Common dimension patterns (e.g., "15×38", "20×40")
            dimension_pattern = re.compile(r'\d+\s*[×x]\s*\d+')
            
            for page_num in page_numbers:
                page = doc[page_num]
                text = page.get_text()
                
                metadata["extracted_text"].append({
                    "page": page_num + 1,
                    "text": text
                })
                
                # Extract dimensions
                dimensions = dimension_pattern.findall(text)
                if dimensions:
                    metadata["dimensions"].extend(dimensions)
                
                # Look for designer mentions
                if "SG NY" in text or "SGNY" in text:
                    if "SG NY" not in metadata["designers"]:
                        metadata["designers"].append("SG NY")
                
                # Look for material keywords
                material_keywords = ["ceramic", "porcelain", "tile", "stone", "marble", "wood"]
                for keyword in material_keywords:
                    if keyword.lower() in text.lower():
                        if keyword not in metadata["materials"]:
                            metadata["materials"].append(keyword)
            
            doc.close()
            
            # Remove duplicates
            metadata["dimensions"] = list(set(metadata["dimensions"]))
            
            self.logger.info(f"✅ Extracted metadata for '{product_name}':")
            self.logger.info(f"   Pages: {metadata['page_numbers']}")
            self.logger.info(f"   Dimensions: {metadata['dimensions']}")
            self.logger.info(f"   Materials: {metadata['materials']}")
            self.logger.info(f"   Designers: {metadata['designers']}")
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata: {e}", exc_info=True)
            raise


def get_focused_product_extractor() -> FocusedProductExtractor:
    """Get singleton instance of FocusedProductExtractor."""
    return FocusedProductExtractor()


