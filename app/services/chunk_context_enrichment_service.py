"""
Chunk Context Enrichment Service - Enhancement 3

Enriches chunks with product context (which product they belong to).
This is a METADATA ENRICHMENT service that runs AFTER chunking, BEFORE storage.

Features:
- Links chunks to products based on page ranges
- Adds product_id and product_name to chunk metadata
- Enables better search filtering and relevance scoring
- Completely optional - doesn't change chunk content

Safety:
- Only adds metadata, doesn't modify chunk content
- Has fallback behavior (if fails, chunks stored without enrichment)
- Feature flag controlled (default: OFF)
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ChunkContextEnrichmentService:
    """
    Enriches chunks with product context metadata.
    
    This service:
    1. Determines which product each chunk belongs to
    2. Adds product_id and product_name to chunk metadata
    3. Enables better search filtering by product
    4. Runs as optional enrichment step before storage
    """
    
    def __init__(self, enabled: bool = False):
        """
        Initialize chunk context enrichment service.
        
        Args:
            enabled: Whether context enrichment is enabled (default: False)
        """
        self.enabled = enabled
        self.logger = logger
    
    async def enrich_chunks(
        self,
        chunks: List[Dict[str, Any]],
        products: List[Dict[str, Any]],
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Enrich chunks with product context.
        
        Args:
            chunks: List of chunks to enrich
            products: List of products with page ranges
            document_id: Document ID for logging
            
        Returns:
            List of enriched chunks (or original chunks if disabled/failed)
        """
        if not self.enabled:
            self.logger.debug("⏭️ Chunk context enrichment disabled (feature flag OFF)")
            return chunks
        
        if not products:
            self.logger.debug("⏭️ No products available for context enrichment")
            return chunks
        
        try:
            self.logger.info(f"🏷️ Enriching {len(chunks)} chunks with product context")
            
            enriched_count = 0
            for chunk in chunks:
                # Get page number from chunk metadata
                page_number = self._get_chunk_page_number(chunk)
                
                if page_number is None:
                    continue
                
                # Find which product this chunk belongs to
                product = self._find_product_for_page(page_number, products)
                
                if product:
                    # ✅ ENRICH: Add product context to metadata
                    if 'metadata' not in chunk:
                        chunk['metadata'] = {}
                    
                    chunk['metadata']['product_id'] = product.get('id')
                    chunk['metadata']['product_name'] = product.get('name')
                    chunk['metadata']['product_page_range'] = product.get('page_range', [])
                    chunk['metadata']['enriched_at'] = datetime.utcnow().isoformat()
                    
                    enriched_count += 1
            
            self.logger.info(f"✅ Enriched {enriched_count}/{len(chunks)} chunks with product context")
            return chunks
            
        except Exception as e:
            self.logger.error(f"❌ Failed to enrich chunks with context: {e}")
            # ✅ FALLBACK: Return original chunks without enrichment
            self.logger.warning("⚠️ Returning chunks without enrichment (fallback)")
            return chunks
    
    def _get_chunk_page_number(self, chunk: Dict[str, Any]) -> Optional[int]:
        """
        Extract page number from chunk metadata.
        
        Args:
            chunk: Chunk dictionary
            
        Returns:
            Page number or None
        """
        try:
            metadata = chunk.get('metadata', {})
            
            # Try different metadata fields
            page_number = (
                metadata.get('page_number') or
                metadata.get('page') or
                metadata.get('page_num')
            )
            
            if page_number is not None:
                return int(page_number)
            
            return None
        except Exception as e:
            self.logger.debug(f"Could not extract page number from chunk: {e}")
            return None
    
    def _find_product_for_page(
        self,
        page_number: int,
        products: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Find which product a page belongs to.
        
        Args:
            page_number: Page number to check
            products: List of products with page ranges
            
        Returns:
            Product dict or None
        """
        for product in products:
            page_range = product.get('page_range', [])
            
            if not page_range or len(page_range) < 2:
                continue
            
            start_page = page_range[0]
            end_page = page_range[1]
            
            # Check if page is within product's range
            if start_page <= page_number <= end_page:
                return product
        
        return None

