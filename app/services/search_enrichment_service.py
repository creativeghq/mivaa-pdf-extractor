"""
Search Enrichment Service

Enriches VECS search results with relationship data from:
- product_image_relationships
- chunk_image_relationships  
- chunk_product_relationships

This service queries the relationship tables to provide complete context
for search results, including related products, chunks, and relevance scores.
"""

import logging
from typing import List, Dict, Any, Optional
from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class SearchEnrichmentService:
    """Service for enriching search results with relationship data."""
    
    def __init__(self):
        self.logger = logger
        self.supabase = get_supabase_client()
    
    async def enrich_image_results(
        self,
        image_results: List[Dict[str, Any]],
        include_products: bool = True,
        include_chunks: bool = True,
        min_relevance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Enrich image search results with related products and chunks.
        
        Args:
            image_results: List of image results from VECS search
            include_products: Whether to include related products
            include_chunks: Whether to include related chunks
            min_relevance: Minimum relevance score to include (0.0-1.0)
            
        Returns:
            Enriched results with products and chunks
        """
        try:
            enriched_results = []
            
            for image_result in image_results:
                image_id = image_result.get('image_id')
                if not image_id:
                    continue
                
                enriched = {
                    **image_result,
                    'related_products': [],
                    'related_chunks': []
                }
                
                # Get related products
                if include_products:
                    products = await self.get_related_products(image_id, min_relevance)
                    enriched['related_products'] = products
                
                # Get related chunks
                if include_chunks:
                    chunks = await self.get_related_chunks(image_id, min_relevance)
                    enriched['related_chunks'] = chunks
                
                enriched_results.append(enriched)
            
            return enriched_results
            
        except Exception as e:
            self.logger.error(f"❌ Failed to enrich image results: {e}")
            return image_results  # Return original results on error
    
    async def get_related_products(
        self,
        image_id: str,
        min_relevance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get products related to an image via product_image_relationships.
        
        Args:
            image_id: Image UUID
            min_relevance: Minimum relevance score
            
        Returns:
            List of related products with relevance scores
        """
        try:
            # Query product_image_relationships with product details
            response = self.supabase.client.table('product_image_relationships')\
                .select('product_id, relevance_score, relationship_type, products(id, name, description, metadata)')\
                .eq('image_id', image_id)\
                .gte('relevance_score', min_relevance)\
                .order('relevance_score', desc=True)\
                .execute()
            
            products = []
            if response.data:
                for rel in response.data:
                    product_data = rel.get('products')
                    if product_data:
                        products.append({
                            'product_id': rel['product_id'],
                            'name': product_data.get('name'),
                            'description': product_data.get('description'),
                            'metadata': product_data.get('metadata', {}),
                            'relevance_score': rel['relevance_score'],
                            'relationship_type': rel['relationship_type']
                        })
            
            return products
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get related products for image {image_id}: {e}")
            return []
    
    async def get_related_chunks(
        self,
        image_id: str,
        min_relevance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Get chunks related to an image via chunk_image_relationships.
        
        Args:
            image_id: Image UUID
            min_relevance: Minimum relevance score
            
        Returns:
            List of related chunks with relevance scores
        """
        try:
            # Query chunk_image_relationships with chunk details
            response = self.supabase.client.table('chunk_image_relationships')\
                .select('chunk_id, relevance_score, relationship_type, document_chunks(id, content, metadata)')\
                .eq('image_id', image_id)\
                .gte('relevance_score', min_relevance)\
                .order('relevance_score', desc=True)\
                .execute()
            
            chunks = []
            if response.data:
                for rel in response.data:
                    chunk_data = rel.get('document_chunks')
                    if chunk_data:
                        chunks.append({
                            'chunk_id': rel['chunk_id'],
                            'content': chunk_data.get('content'),
                            'metadata': chunk_data.get('metadata', {}),
                            'relevance_score': rel['relevance_score'],
                            'relationship_type': rel['relationship_type']
                        })
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get related chunks for image {image_id}: {e}")
            return []

