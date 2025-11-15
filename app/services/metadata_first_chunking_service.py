"""
Metadata-First Chunking Service - Enhancement 4

This service implements metadata-first architecture:
1. Extract product metadata FIRST (before chunking)
2. Identify which pages contain product metadata
3. Exclude those pages from chunking to avoid duplication
4. Result: Zero duplication between chunks and metadata

Feature Flag: ENABLE_METADATA_FIRST (default: False)
"""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)


class MetadataFirstChunkingService:
    """
    Service for metadata-first chunking architecture.
    
    This service ensures that product metadata pages are excluded from
    chunking to prevent duplication between chunks and product metadata.
    """
    
    def __init__(self, enabled: bool = False):
        """
        Initialize metadata-first chunking service.
        
        Args:
            enabled: Whether this service is enabled (default: False)
        """
        self.enabled = enabled
        self.logger = logger
        
        if self.enabled:
            self.logger.info("✅ Metadata-First Chunking Service ENABLED")
        else:
            self.logger.info("⚪ Metadata-First Chunking Service DISABLED")
    
    async def get_pages_to_exclude(
        self,
        products: List[Dict[str, Any]],
        document_id: str
    ) -> Set[int]:
        """
        Get set of page numbers to exclude from chunking.
        
        These are pages that contain product metadata and should not
        be chunked to avoid duplication.
        
        Args:
            products: List of products with page_range metadata
            document_id: Document ID for logging
            
        Returns:
            Set of page numbers to exclude from chunking
        """
        if not self.enabled:
            return set()  # Don't exclude any pages if disabled
        
        try:
            excluded_pages = set()
            
            for product in products:
                # Get page range for this product
                page_range = product.get('page_range', [])
                
                if not page_range or len(page_range) < 2:
                    continue
                
                # Add all pages in the product's range to exclusion set
                start_page = page_range[0]
                end_page = page_range[1]
                
                for page_num in range(start_page, end_page + 1):
                    excluded_pages.add(page_num)
                
                self.logger.debug(
                    f"   📄 Product '{product.get('name')}': "
                    f"excluding pages {start_page}-{end_page}"
                )
            
            if excluded_pages:
                self.logger.info(
                    f"🚫 Metadata-First: Excluding {len(excluded_pages)} pages "
                    f"from chunking (product metadata pages)"
                )
                self.logger.info(f"   Pages to exclude: {sorted(excluded_pages)}")
            
            return excluded_pages
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get pages to exclude: {e}")
            return set()  # Fallback: don't exclude any pages
    
    async def filter_text_for_chunking(
        self,
        full_text: str,
        excluded_pages: Set[int],
        page_delimiter: str = "\n---PAGE---\n"
    ) -> str:
        """
        Filter text to exclude metadata pages before chunking.
        
        Args:
            full_text: Full PDF text with page delimiters
            excluded_pages: Set of page numbers to exclude
            page_delimiter: Delimiter used to separate pages in text
            
        Returns:
            Filtered text with metadata pages removed
        """
        if not self.enabled or not excluded_pages:
            return full_text  # Return full text if disabled or no exclusions
        
        try:
            # Split text by page delimiter
            pages = full_text.split(page_delimiter)
            
            # Filter out excluded pages (1-indexed)
            filtered_pages = []
            for page_num, page_text in enumerate(pages, start=1):
                if page_num not in excluded_pages:
                    filtered_pages.append(page_text)
            
            # Rejoin filtered pages
            filtered_text = page_delimiter.join(filtered_pages)
            
            original_length = len(full_text)
            filtered_length = len(filtered_text)
            reduction_pct = ((original_length - filtered_length) / original_length * 100) if original_length > 0 else 0
            
            self.logger.info(
                f"✅ Metadata-First: Filtered text from {original_length:,} to {filtered_length:,} characters "
                f"({reduction_pct:.1f}% reduction)"
            )
            
            return filtered_text
            
        except Exception as e:
            self.logger.error(f"❌ Failed to filter text: {e}")
            return full_text  # Fallback: return original text
    
    async def process_for_metadata_first(
        self,
        full_text: str,
        products: List[Dict[str, Any]],
        document_id: str
    ) -> Dict[str, Any]:
        """
        Main processing method for metadata-first architecture.
        
        Args:
            full_text: Full PDF text
            products: List of products with page ranges
            document_id: Document ID
            
        Returns:
            Dict with filtered_text and metadata
        """
        if not self.enabled:
            return {
                'filtered_text': full_text,
                'excluded_pages': set(),
                'metadata': {'metadata_first_enabled': False}
            }
        
        try:
            # Get pages to exclude
            excluded_pages = await self.get_pages_to_exclude(products, document_id)
            
            # Filter text
            filtered_text = await self.filter_text_for_chunking(full_text, excluded_pages)
            
            return {
                'filtered_text': filtered_text,
                'excluded_pages': excluded_pages,
                'metadata': {
                    'metadata_first_enabled': True,
                    'excluded_page_count': len(excluded_pages),
                    'excluded_pages': sorted(excluded_pages),
                    'original_text_length': len(full_text),
                    'filtered_text_length': len(filtered_text)
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Metadata-first processing failed: {e}")
            # Fallback: return original text
            return {
                'filtered_text': full_text,
                'excluded_pages': set(),
                'metadata': {'metadata_first_enabled': False, 'error': str(e)}
            }

