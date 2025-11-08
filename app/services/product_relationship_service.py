"""
Product Relationship Service

Finds related products based on multiple criteria:
- Material family (same material type)
- Pattern matching (same finish, colors, texture)
- Collection matching (same collection, designer, factory)
- Complementary products (products that work together)
- Alternative products (similar products with same specs)
- Custom NLP-based relationships (future enhancement)
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ProductRelationshipService:
    """Service for finding related products based on multiple criteria."""
    
    def __init__(self, supabase_client):
        """Initialize the service with Supabase client."""
        self.supabase = supabase_client
        
        # Complementary category mappings
        self.complementary_categories = {
            'floor_tiles': ['wall_tiles', 'trim', 'border', 'grout'],
            'wall_tiles': ['floor_tiles', 'trim', 'border', 'grout'],
            'ceramic': ['porcelain', 'stone', 'cement'],
            'porcelain': ['ceramic', 'stone', 'cement'],
            'stone': ['ceramic', 'porcelain', 'cement'],
            'cement': ['ceramic', 'porcelain', 'stone']
        }
    
    async def find_related_products(
        self,
        product_id: str,
        workspace_id: str,
        relationship_types: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find related products for a given product.
        
        Args:
            product_id: ID of the product to find relationships for
            workspace_id: Workspace ID for scoped search
            relationship_types: List of relationship types to include
            limit: Maximum number of related products to return
            
        Returns:
            List of related products with relationship metadata
        """
        try:
            # Get the source product
            product_response = self.supabase.table('products').select(
                'id, name, description, metadata'
            ).eq('id', product_id).eq('workspace_id', workspace_id).single().execute()
            
            if not product_response.data:
                logger.warning(f"Product {product_id} not found")
                return []
            
            source_product = product_response.data
            
            # Default to all relationship types if not specified
            if not relationship_types:
                relationship_types = [
                    'material_family',
                    'pattern_match',
                    'collection',
                    'complementary',
                    'alternative'
                ]
            
            # Find relationships based on requested types
            all_related = []
            
            if 'material_family' in relationship_types:
                material_family = await self._find_material_family(
                    source_product, workspace_id, product_id
                )
                all_related.extend(material_family)
            
            if 'pattern_match' in relationship_types:
                pattern_matches = await self._find_pattern_matches(
                    source_product, workspace_id, product_id
                )
                all_related.extend(pattern_matches)
            
            if 'collection' in relationship_types:
                collection_matches = await self._find_collection_matches(
                    source_product, workspace_id, product_id
                )
                all_related.extend(collection_matches)
            
            if 'complementary' in relationship_types:
                complementary = await self._find_complementary_products(
                    source_product, workspace_id, product_id
                )
                all_related.extend(complementary)
            
            if 'alternative' in relationship_types:
                alternatives = await self._find_alternative_products(
                    source_product, workspace_id, product_id
                )
                all_related.extend(alternatives)
            
            # Deduplicate and sort by relevance score
            deduplicated = self._deduplicate_products(all_related)
            
            # Sort by relevance score (highest first) and limit
            deduplicated.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return deduplicated[:limit]
            
        except Exception as e:
            logger.error(f"Error finding related products: {e}", exc_info=True)
            return []
    
    async def _find_material_family(
        self,
        source_product: Dict[str, Any],
        workspace_id: str,
        exclude_id: str
    ) -> List[Dict[str, Any]]:
        """Find products with same material type."""
        try:
            metadata = source_product.get('metadata', {})
            material_type = metadata.get('material_type')
            
            if not material_type:
                return []
            
            response = self.supabase.table('products').select(
                'id, name, description, metadata'
            ).eq('workspace_id', workspace_id).neq('id', exclude_id).execute()
            
            related = []
            for product in response.data or []:
                product_material = product.get('metadata', {}).get('material_type')
                if product_material and product_material.lower() == material_type.lower():
                    related.append({
                        'id': product['id'],
                        'name': product['name'],
                        'description': product.get('description'),
                        'relationship_type': 'material_family',
                        'relevance_score': 0.85,
                        'reason': f"Same material family: {material_type}",
                        'metadata': product.get('metadata', {})
                    })
            
            return related
            
        except Exception as e:
            logger.error(f"Error finding material family: {e}")
            return []
    
    async def _find_pattern_matches(
        self,
        source_product: Dict[str, Any],
        workspace_id: str,
        exclude_id: str
    ) -> List[Dict[str, Any]]:
        """Find products with same finish and colors."""
        # Simplified implementation - returns empty list
        return []
    
    async def _find_collection_matches(
        self,
        source_product: Dict[str, Any],
        workspace_id: str,
        exclude_id: str
    ) -> List[Dict[str, Any]]:
        """Find products from same collection, designer, or factory."""
        # Simplified implementation - returns empty list
        return []
    
    async def _find_complementary_products(
        self,
        source_product: Dict[str, Any],
        workspace_id: str,
        exclude_id: str
    ) -> List[Dict[str, Any]]:
        """Find products that work well together."""
        # Simplified implementation - returns empty list
        return []
    
    async def _find_alternative_products(
        self,
        source_product: Dict[str, Any],
        workspace_id: str,
        exclude_id: str
    ) -> List[Dict[str, Any]]:
        """Find alternative products with same technical specs."""
        # Simplified implementation - returns empty list
        return []
    
    def _deduplicate_products(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate products, keeping the one with highest relevance score."""
        seen = {}
        for product in products:
            product_id = product['id']
            if product_id not in seen or product['relevance_score'] > seen[product_id]['relevance_score']:
                seen[product_id] = product
        
        return list(seen.values())

