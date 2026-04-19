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
import json
from typing import List, Dict, Any, Optional
from app.config import get_settings
from app.services.core.ai_client_service import get_ai_client_service

logger = logging.getLogger(__name__)


class ProductRelationshipService:
    """Service for finding related products based on multiple criteria."""
    
    def __init__(self, supabase_client):
        """Initialize the service with Supabase client."""
        self.supabase = supabase_client
        self.settings = get_settings()
        
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
        limit: int = 5,
        custom_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find related products for a given product.
        
        Args:
            product_id: ID of the product to find relationships for
            workspace_id: Workspace ID for scoped search
            relationship_types: List of relationship types to include
                               (material_family, pattern_match, collection, complementary, alternative, custom)
            limit: Maximum number of related products to return
            custom_prompt: Optional NLP prompt for custom relationships
            
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
            
            if 'custom' in relationship_types and custom_prompt:
                custom_related = await self._find_custom_relationships(
                    source_product, workspace_id, product_id, custom_prompt
                )
                all_related.extend(custom_related)
            
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
        try:
            metadata = source_product.get('metadata', {})
            finish = metadata.get('finish')
            colors = metadata.get('colors', [])
            
            if not finish and not colors:
                return []
            
            response = self.supabase.table('products').select(
                'id, name, description, metadata'
            ).eq('workspace_id', workspace_id).neq('id', exclude_id).execute()
            
            related = []
            for product in response.data or []:
                product_metadata = product.get('metadata', {})
                product_finish = product_metadata.get('finish')
                product_colors = product_metadata.get('colors', [])
                
                score = 0.0
                reasons = []
                
                # Check finish match
                if finish and product_finish and finish.lower() == product_finish.lower():
                    score += 0.5
                    reasons.append(f"Same finish: {finish}")
                
                # Check color overlap
                if colors and product_colors:
                    common_colors = set(c.lower() for c in colors) & set(c.lower() for c in product_colors)
                    if common_colors:
                        color_score = len(common_colors) / max(len(colors), len(product_colors))
                        score += color_score * 0.35
                        reasons.append(f"Matching colors: {', '.join(common_colors)}")
                
                if score > 0.3:  # Minimum threshold
                    related.append({
                        'id': product['id'],
                        'name': product['name'],
                        'description': product.get('description'),
                        'relationship_type': 'pattern_match',
                        'relevance_score': min(0.85, 0.75 + score * 0.1),
                        'reason': '; '.join(reasons),
                        'metadata': product_metadata
                    })
            
            return related

        except Exception as e:
            logger.error(f"Error finding pattern matches: {e}")
            return []

    async def _find_collection_matches(
        self,
        source_product: Dict[str, Any],
        workspace_id: str,
        exclude_id: str
    ) -> List[Dict[str, Any]]:
        """Find products from same collection, designer, or factory."""
        try:
            metadata = source_product.get('metadata', {})
            collection = metadata.get('collection')
            designer = metadata.get('designer')
            factory_name = metadata.get('factory_name')

            if not collection and not designer and not factory_name:
                return []

            response = self.supabase.table('products').select(
                'id, name, description, metadata'
            ).eq('workspace_id', workspace_id).neq('id', exclude_id).execute()

            related = []
            for product in response.data or []:
                product_metadata = product.get('metadata', {})
                product_collection = product_metadata.get('collection')
                product_designer = product_metadata.get('designer')
                product_factory = product_metadata.get('factory_name')

                score = 0.0
                reason = None

                # Collection match (highest priority)
                if collection and product_collection and collection.lower() == product_collection.lower():
                    score = 0.90
                    reason = f"Same collection: {collection}"
                # Designer match (medium priority)
                elif designer and product_designer and designer.lower() == product_designer.lower():
                    score = 0.80
                    reason = f"Same designer: {designer}"
                # Factory match (lower priority)
                elif factory_name and product_factory and factory_name.lower() == product_factory.lower():
                    score = 0.70
                    reason = f"Same factory: {factory_name}"

                if score > 0:
                    related.append({
                        'id': product['id'],
                        'name': product['name'],
                        'description': product.get('description'),
                        'relationship_type': 'collection',
                        'relevance_score': score,
                        'reason': reason,
                        'metadata': product_metadata
                    })

            return related

        except Exception as e:
            logger.error(f"Error finding collection matches: {e}")
            return []

    async def _find_complementary_products(
        self,
        source_product: Dict[str, Any],
        workspace_id: str,
        exclude_id: str
    ) -> List[Dict[str, Any]]:
        """Find products that work well together."""
        try:
            metadata = source_product.get('metadata', {})
            category = metadata.get('category')
            material_type = metadata.get('material_type')

            if not category and not material_type:
                return []

            # Get complementary categories
            complementary_cats = []
            if category:
                complementary_cats.extend(self.complementary_categories.get(category.lower(), []))
            if material_type:
                complementary_cats.extend(self.complementary_categories.get(material_type.lower(), []))

            if not complementary_cats:
                return []

            response = self.supabase.table('products').select(
                'id, name, description, metadata'
            ).eq('workspace_id', workspace_id).neq('id', exclude_id).execute()

            related = []
            for product in response.data or []:
                product_metadata = product.get('metadata', {})
                product_category = product_metadata.get('category', '').lower()
                product_material = product_metadata.get('material_type', '').lower()

                if product_category in complementary_cats or product_material in complementary_cats:
                    related.append({
                        'id': product['id'],
                        'name': product['name'],
                        'description': product.get('description'),
                        'relationship_type': 'complementary',
                        'relevance_score': 0.75,
                        'reason': f"Complements {category or material_type}",
                        'metadata': product_metadata
                    })

            return related

        except Exception as e:
            logger.error(f"Error finding complementary products: {e}")
            return []

    async def _find_alternative_products(
        self,
        source_product: Dict[str, Any],
        workspace_id: str,
        exclude_id: str
    ) -> List[Dict[str, Any]]:
        """Find alternative products with same technical specs."""
        try:
            metadata = source_product.get('metadata', {})
            slip_resistance = metadata.get('slip_resistance')
            fire_rating = metadata.get('fire_rating')
            dimensions = metadata.get('dimensions', [])

            if not slip_resistance and not fire_rating and not dimensions:
                return []

            response = self.supabase.table('products').select(
                'id, name, description, metadata'
            ).eq('workspace_id', workspace_id).neq('id', exclude_id).execute()

            related = []
            for product in response.data or []:
                product_metadata = product.get('metadata', {})
                product_slip = product_metadata.get('slip_resistance')
                product_fire = product_metadata.get('fire_rating')
                product_dims = product_metadata.get('dimensions', [])

                score = 0.0
                reasons = []

                # Check slip resistance match
                if slip_resistance and product_slip and slip_resistance == product_slip:
                    score += 0.4
                    reasons.append(f"Same slip resistance: {slip_resistance}")

                # Check fire rating match
                if fire_rating and product_fire and fire_rating == product_fire:
                    score += 0.3
                    reasons.append(f"Same fire rating: {fire_rating}")

                # Check dimension overlap
                if dimensions and product_dims:
                    common_dims = set(dimensions) & set(product_dims)
                    if common_dims:
                        score += 0.3
                        reasons.append(f"Matching dimensions: {', '.join(common_dims)}")

                if score >= 0.3:  # Minimum threshold
                    related.append({
                        'id': product['id'],
                        'name': product['name'],
                        'description': product.get('description'),
                        'relationship_type': 'alternative',
                        'relevance_score': min(0.70, 0.60 + score * 0.1),
                        'reason': '; '.join(reasons),
                        'metadata': product_metadata
                    })

            return related

        except Exception as e:
            logger.error(f"Error finding alternative products: {e}")
            return []

    async def _find_custom_relationships(
        self,
        source_product: Dict[str, Any],
        workspace_id: str,
        exclude_id: str,
        custom_prompt: str
    ) -> List[Dict[str, Any]]:
        """Find products using a custom LLM-evaluated prompt against the product catalog."""
        try:
            # Fetch candidate products from workspace
            response = self.supabase.table("products").select(
                "id, name, description, material_type, color, finish, collection"
            ).eq("workspace_id", workspace_id).neq("id", exclude_id).limit(50).execute()

            if not response.data:
                return []

            candidates = response.data

            # Ask Claude to select related products based on the custom prompt
            ai = get_ai_client_service()
            source_summary = {
                "name": source_product.get("name", ""),
                "description": source_product.get("description", ""),
                "material_type": source_product.get("material_type", ""),
                "color": source_product.get("color", ""),
                "finish": source_product.get("finish", ""),
                "collection": source_product.get("collection", ""),
            }
            candidate_list = [
                {"index": i, "id": c["id"], "name": c.get("name", ""),
                 "material_type": c.get("material_type", ""), "color": c.get("color", ""),
                 "finish": c.get("finish", "")}
                for i, c in enumerate(candidates)
            ]

            prompt = (
                f"{custom_prompt}\n\n"
                f"Source product: {json.dumps(source_summary)}\n\n"
                f"Candidate products: {json.dumps(candidate_list)}\n\n"
                "Return a JSON array of objects with 'index' and 'relevance_score' (0.0-1.0) "
                "for candidates that match the custom relationship criteria. "
                "Include only matches with score >= 0.5. "
                "Example: [{\"index\": 2, \"relevance_score\": 0.85}]. "
                "Return ONLY the JSON array."
            )

            from app.services.core.claude_helper import tracked_claude_call_async
            resp = await tracked_claude_call_async(
                task="product_relationship_detection",
                model="claude-haiku-4-5",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            matches = json.loads(resp.content[0].text.strip())

            related = []
            for match in matches:
                idx = match.get("index")
                score = float(match.get("relevance_score", 0.5))
                if idx is not None and idx < len(candidates):
                    product = candidates[idx]
                    related.append({
                        **product,
                        "relevance_score": score,
                        "relationship_type": "custom",
                    })

            logger.info(f"Custom relationship detection found {len(related)} matches")
            return related

        except Exception as e:
            logger.error(f"Error in custom relationship detection: {e}")
            return []

    def _deduplicate_products(self, products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate products, keeping the one with highest relevance score."""
        seen = {}
        for product in products:
            product_id = product['id']
            if product_id not in seen or product['relevance_score'] > seen[product_id]['relevance_score']:
                seen[product_id] = product

        return list(seen.values())


