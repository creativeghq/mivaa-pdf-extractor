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
        min_relevance: float = 0.0,
        rerank: bool = True,
        # ✅ UPDATED: Support multi-vector scoring with all 6 embedding types
        visual_weight: float = 0.30,
        relevance_weight: float = 0.20,
        color_weight: float = 0.125,
        texture_weight: float = 0.125,
        style_weight: float = 0.125,
        material_weight: float = 0.125
    ) -> List[Dict[str, Any]]:
        """
        Enrich image search results with related products and chunks.
        Optionally re-rank results by combining ALL embedding types + relationship relevance.

        Args:
            image_results: List of image results from VECS search
            include_products: Whether to include related products
            include_chunks: Whether to include related chunks
            min_relevance: Minimum relevance score to include (0.0-1.0)
            rerank: Whether to re-rank results by combined score (default: True)
            visual_weight: Weight for visual similarity score (default: 0.30)
            relevance_weight: Weight for relationship relevance score (default: 0.20)
            color_weight: Weight for color similarity score (default: 0.125)
            texture_weight: Weight for texture similarity score (default: 0.125)
            style_weight: Weight for style similarity score (default: 0.125)
            material_weight: Weight for material similarity score (default: 0.125)

        Returns:
            Enriched results with products and chunks, optionally re-ranked

        Note:
            Total weights = 1.0 (30% visual + 20% relevance + 50% specialized embeddings)
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

                    # ✅ UPDATED: Multi-vector combined score with all 6 embedding types
                    if rerank and products:
                        # Use highest product relevance score for re-ranking
                        max_product_relevance = max(p.get('relevance_score', 0.0) for p in products)

                        # Get all similarity scores from metadata
                        metadata = image_result.get('metadata', {})
                        embedding_scores = metadata.get('embedding_scores', {})

                        # Extract individual scores (fallback to similarity_score if not in metadata)
                        visual_sim = embedding_scores.get('visual', image_result.get('similarity_score', 0.0))
                        color_sim = embedding_scores.get('color', 0.0)
                        texture_sim = embedding_scores.get('texture', 0.0)
                        style_sim = embedding_scores.get('style', 0.0)
                        material_sim = embedding_scores.get('material', 0.0)

                        # Combined score: weighted average of ALL embedding types + product relevance
                        combined_score = (
                            visual_weight * visual_sim +
                            relevance_weight * max_product_relevance +
                            color_weight * color_sim +
                            texture_weight * texture_sim +
                            style_weight * style_sim +
                            material_weight * material_sim
                        )

                        enriched['combined_score'] = combined_score
                        enriched['max_product_relevance'] = max_product_relevance

                        # ✅ NEW: Add score breakdown for debugging and transparency
                        enriched['score_breakdown'] = {
                            'visual': visual_sim,
                            'relevance': max_product_relevance,
                            'color': color_sim,
                            'texture': texture_sim,
                            'style': style_sim,
                            'material': material_sim,
                            'weights': {
                                'visual': visual_weight,
                                'relevance': relevance_weight,
                                'color': color_weight,
                                'texture': texture_weight,
                                'style': style_weight,
                                'material': material_weight
                            }
                        }
                    else:
                        # No products or rerank disabled - use visual similarity only
                        enriched['combined_score'] = image_result.get('similarity_score', 0.0)
                        enriched['max_product_relevance'] = 0.0
                        enriched['score_breakdown'] = {
                            'visual': image_result.get('similarity_score', 0.0),
                            'relevance': 0.0,
                            'color': 0.0,
                            'texture': 0.0,
                            'style': 0.0,
                            'material': 0.0
                        }

                # Get related chunks
                if include_chunks:
                    chunks = await self.get_related_chunks(image_id, min_relevance)
                    enriched['related_chunks'] = chunks

                enriched_results.append(enriched)

            # ✅ UPDATED: Re-rank by multi-vector combined score if enabled
            if rerank:
                enriched_results.sort(key=lambda x: x.get('combined_score', 0.0), reverse=True)
                self.logger.info(
                    f"✅ Re-ranked {len(enriched_results)} results by multi-vector score "
                    f"(visual {visual_weight} + relevance {relevance_weight} + "
                    f"color {color_weight} + texture {texture_weight} + "
                    f"style {style_weight} + material {material_weight})"
                )

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

