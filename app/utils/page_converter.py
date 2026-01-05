"""
Page Number Conversion Utility

Centralizes all page number conversions to prevent catalog/PDF/index mismatches.

PROBLEM:
- Catalog pages (1-based, logical): "This product is on catalog page 74"
- PDF pages (1-based, physical): "Open page 37 in the PDF viewer"
- Array indices (0-based, technical): "Access doc[36] in PyMuPDF"
- 2-page spreads: Catalog page 74 might be on PDF page 37

SOLUTION:
- Single source of truth for all conversions
- Type-safe page number objects
- Automatic detection of spread layout
"""

from dataclasses import dataclass
from typing import Optional, List
import fitz  # PyMuPDF


@dataclass(frozen=True)
class PageNumber:
    """
    Immutable page number with clear semantics.
    
    Each instance represents ONE specific location in a document,
    expressed in all three numbering systems simultaneously.
    """
    catalog_page: int      # Human-readable catalog page (1-based)
    pdf_page: int          # Physical PDF page (1-based)
    array_index: int       # Python array index (0-based)
    pages_per_sheet: int   # Layout: 1 = standard, 2 = spread
    
    def __post_init__(self):
        """Validate page numbers are consistent."""
        # Validate catalog_page
        if self.catalog_page < 1:
            raise ValueError(f"Catalog page must be >= 1, got {self.catalog_page}")
        
        # Validate pdf_page
        if self.pdf_page < 1:
            raise ValueError(f"PDF page must be >= 1, got {self.pdf_page}")
        
        # Validate array_index
        if self.array_index < 0:
            raise ValueError(f"Array index must be >= 0, got {self.array_index}")
        
        # Validate pages_per_sheet
        if self.pages_per_sheet not in [1, 2]:
            raise ValueError(f"Pages per sheet must be 1 or 2, got {self.pages_per_sheet}")
        
        # Validate consistency: pdf_page should equal array_index + 1
        if self.pdf_page != self.array_index + 1:
            raise ValueError(
                f"Inconsistent page numbers: pdf_page={self.pdf_page}, "
                f"array_index={self.array_index} (should be {self.pdf_page - 1})"
            )
        
        # Validate catalog-to-PDF conversion
        expected_pdf_page = (self.catalog_page + self.pages_per_sheet - 1) // self.pages_per_sheet
        if self.pdf_page != expected_pdf_page:
            raise ValueError(
                f"Inconsistent conversion: catalog_page={self.catalog_page}, "
                f"pages_per_sheet={self.pages_per_sheet}, "
                f"pdf_page={self.pdf_page} (expected {expected_pdf_page})"
            )
    
    def __str__(self):
        """Human-readable representation."""
        if self.pages_per_sheet == 2:
            return f"catalog:{self.catalog_page} → PDF:{self.pdf_page} (spread) → idx:{self.array_index}"
        else:
            return f"catalog:{self.catalog_page} → PDF:{self.pdf_page} → idx:{self.array_index}"
    
    def __repr__(self):
        return f"PageNumber({self.catalog_page}, {self.pdf_page}, {self.array_index}, pps={self.pages_per_sheet})"


class PageConverter:
    """
    Centralized page number conversion manager.
    
    Usage:
        converter = PageConverter.from_pdf_path("catalog.pdf")
        
        # Convert catalog page to all formats
        page = converter.from_catalog_page(74)
        print(page.pdf_page)      # 37 (if spreads)
        print(page.array_index)   # 36
        
        # Convert PDF page to all formats
        page = converter.from_pdf_page(37)
        print(page.catalog_page)  # 73 or 74 (depends on spread)
        
        # Convert array index to all formats
        page = converter.from_array_index(36)
        print(page.catalog_page)  # 73 or 74 (depends on spread)
    """
    
    def __init__(self, pages_per_sheet: int = 1, total_pdf_pages: Optional[int] = None):
        """
        Initialize page converter.
        
        Args:
            pages_per_sheet: 1 for standard layout, 2 for 2-page spreads
            total_pdf_pages: Total number of physical pages in PDF (for validation)
        """
        if pages_per_sheet not in [1, 2]:
            raise ValueError(f"pages_per_sheet must be 1 or 2, got {pages_per_sheet}")
        
        self.pages_per_sheet = pages_per_sheet
        self.total_pdf_pages = total_pdf_pages
        self.total_catalog_pages = total_pdf_pages * pages_per_sheet if total_pdf_pages else None
    
    @classmethod
    def from_pdf_path(cls, pdf_path: str, pages_to_check: int = 5) -> 'PageConverter':
        """
        Auto-detect layout from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            pages_to_check: Number of pages to check for layout detection
        
        Returns:
            PageConverter with detected layout
        """
        doc = fitz.open(pdf_path)
        total_pdf_pages = len(doc)
        
        # Auto-detect 2-page spreads
        pages_per_sheet = 1  # Default
        
        if total_pdf_pages > 0:
            spread_count = 0
            standard_count = 0
            
            for page_idx in range(min(pages_to_check, total_pdf_pages)):
                page = doc[page_idx]
                rect = page.rect
                aspect_ratio = rect.width / rect.height if rect.height > 0 else 1.0
                
                # Landscape pages with aspect ratio > 1.4 are likely 2-page spreads
                if aspect_ratio > 1.4:
                    spread_count += 1
                else:
                    standard_count += 1
            
            # Use majority vote
            if spread_count > standard_count:
                pages_per_sheet = 2
        
        doc.close()
        
        return cls(pages_per_sheet=pages_per_sheet, total_pdf_pages=total_pdf_pages)
    
    def from_catalog_page(self, catalog_page: int) -> PageNumber:
        """
        Convert catalog page (1-based) to all formats.
        
        Args:
            catalog_page: Catalog page number (1-based)
        
        Returns:
            PageNumber object with all conversions
        """
        if catalog_page < 1:
            raise ValueError(f"Catalog page must be >= 1, got {catalog_page}")
        
        # Calculate PDF page using the formula
        pdf_page = (catalog_page + self.pages_per_sheet - 1) // self.pages_per_sheet
        
        # Validate against total pages
        if self.total_pdf_pages and pdf_page > self.total_pdf_pages:
            raise ValueError(
                f"Catalog page {catalog_page} maps to PDF page {pdf_page}, "
                f"but PDF only has {self.total_pdf_pages} pages"
            )
        
        # Calculate array index (0-based)
        array_index = pdf_page - 1
        
        return PageNumber(
            catalog_page=catalog_page,
            pdf_page=pdf_page,
            array_index=array_index,
            pages_per_sheet=self.pages_per_sheet
        )
    
    def from_pdf_page(self, pdf_page: int) -> PageNumber:
        """
        Convert PDF page (1-based) to all formats.
        
        For spreads, returns the FIRST catalog page on this PDF page.
        Example: PDF page 37 → catalog page 73 (if 2-page spreads)
        
        Args:
            pdf_page: PDF page number (1-based)
        
        Returns:
            PageNumber object with all conversions
        """
        if pdf_page < 1:
            raise ValueError(f"PDF page must be >= 1, got {pdf_page}")
        
        # Validate against total pages
        if self.total_pdf_pages and pdf_page > self.total_pdf_pages:
            raise ValueError(
                f"PDF page {pdf_page} exceeds total pages {self.total_pdf_pages}"
            )
        
        # Calculate first catalog page on this PDF page
        # For spreads: PDF page 37 → catalog pages 73-74 → return 73
        catalog_page = (pdf_page - 1) * self.pages_per_sheet + 1
        
        # Calculate array index
        array_index = pdf_page - 1
        
        return PageNumber(
            catalog_page=catalog_page,
            pdf_page=pdf_page,
            array_index=array_index,
            pages_per_sheet=self.pages_per_sheet
        )
    
    def from_array_index(self, array_index: int) -> PageNumber:
        """
        Convert array index (0-based) to all formats.
        
        Args:
            array_index: Array index (0-based)
        
        Returns:
            PageNumber object with all conversions
        """
        if array_index < 0:
            raise ValueError(f"Array index must be >= 0, got {array_index}")
        
        # Validate against total pages
        if self.total_pdf_pages and array_index >= self.total_pdf_pages:
            raise ValueError(
                f"Array index {array_index} exceeds total pages {self.total_pdf_pages}"
            )
        
        # Convert to PDF page first
        pdf_page = array_index + 1
        
        # Then use from_pdf_page for consistency
        return self.from_pdf_page(pdf_page)
    
    def validate_page_range(self, catalog_pages: List[int]) -> List[int]:
        """
        Validate catalog page range and filter out-of-bounds pages.
        
        Args:
            catalog_pages: List of catalog page numbers (1-based)
        
        Returns:
            List of valid catalog pages (out-of-bounds pages removed)
        """
        valid_pages = []
        invalid_pages = []
        
        for catalog_page in catalog_pages:
            try:
                page = self.from_catalog_page(catalog_page)
                valid_pages.append(catalog_page)
            except ValueError:
                invalid_pages.append(catalog_page)
        
        if invalid_pages:
            # Log warning (but don't raise - just filter)
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(
                f"Filtered {len(invalid_pages)} out-of-bounds catalog pages: "
                f"{invalid_pages[:5]}{'...' if len(invalid_pages) > 5 else ''} "
                f"(PDF has {self.total_pdf_pages} pages = "
                f"{self.total_catalog_pages} catalog pages)"
            )
        
        return valid_pages
    
    def get_catalog_pages_on_pdf_page(self, pdf_page: int) -> List[int]:
        """
        Get all catalog pages on a given PDF page.
        
        For standard layout: Returns [catalog_page]
        For 2-page spreads: Returns [catalog_page, catalog_page + 1]
        
        Args:
            pdf_page: PDF page number (1-based)
        
        Returns:
            List of catalog page numbers on this PDF page
        """
        first_catalog_page = self.from_pdf_page(pdf_page).catalog_page
        
        if self.pages_per_sheet == 1:
            return [first_catalog_page]
        else:  # 2-page spreads
            return [first_catalog_page, first_catalog_page + 1]


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """Demonstrate how to use PageConverter."""
    
    # Example 1: Auto-detect from PDF
    converter = PageConverter.from_pdf_path("/path/to/catalog.pdf")
    print(f"Detected layout: {converter.pages_per_sheet} pages per sheet")
    print(f"Total PDF pages: {converter.total_pdf_pages}")
    print(f"Total catalog pages: {converter.total_catalog_pages}")
    
    # Example 2: Convert catalog page to PDF page / index
    page = converter.from_catalog_page(74)
    print(f"\nCatalog page 74:")
    print(f"  PDF page: {page.pdf_page}")
    print(f"  Array index: {page.array_index}")
    print(f"  {page}")  # Pretty print
    
    # Example 3: Convert PDF page to catalog page
    page = converter.from_pdf_page(37)
    print(f"\nPDF page 37:")
    print(f"  First catalog page: {page.catalog_page}")
    print(f"  All catalog pages: {converter.get_catalog_pages_on_pdf_page(37)}")
    
    # Example 4: Validate page range
    product_pages = [1, 2, 3, 74, 200, 500]  # Some invalid
    valid_pages = converter.validate_page_range(product_pages)
    print(f"\nValidated pages: {valid_pages}")
    
    # Example 5: Use in vision extraction
    for catalog_page in valid_pages:
        page = converter.from_catalog_page(catalog_page)
        print(f"  Extract from PDF page {page.pdf_page} (index {page.array_index})")


if __name__ == "__main__":
    example_usage()
