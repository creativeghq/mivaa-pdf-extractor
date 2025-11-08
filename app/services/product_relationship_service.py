"""
Product Relationship Service

Handles finding related products based on:
1. Material family (same material type, can be combined)
2. Pattern matching (same design patterns)
3. NLP-based customization (admin-defined relationship rules)
4. Metadata similarity (factory, designer, collection, etc.)
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ProductRelationshipService:
    """
    Service for finding related products based on multiple criteria.
    
    Relationship Types:
    - material_family: Same material type, can be combined/matched
    - pattern_match: Same design pattern or visual style
    - collection: Same collection or designer
    - complementary: Products that work well together
    - alternative: Similar product with different properties
    - custom: Admin-defined relationship via NLP rules
    """

    def __init__(self, supabase_client=None, llm_service=None):
        """
        Initialize the product relationship service.
        
        Args:
            supabase_client: Supabase client for database queries
            llm_service: LLM service for NLP-based relationship detection
        """
        self.supabase = supabase_client
        self.llm = llm_service

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
            product_id: ID of the product to find relations for
            workspace_id: Workspace ID for scoped search
            relationship_types: Types of relationships to find
                (material_family, pattern_match, collection, complementary, alternative, custom)
            limit: Maximum number of related products to return
            custom_prompt: Optional NLP prompt for custom relationship detection
            
        Returns:
            List of related products with relationship metadata
        """
        if not self.supabase:
            logger.warning("Supabase client not available for product relationships")
            return []

        try:
            # Get the source product
            source_product = await self._get_product(product_id, workspace_id)
            if not source_product:
                logger.warning(f"Product {product_id} not found")
                return []

            related_products = []

            # Default relationship types if not specified
            if not relationship_types:
                relationship_types = [
                    "material_family",
                    "pattern_match",
                    "collection",
                    "complementary"
                ]

            # Find products by each relationship type
            for rel_type in relationship_types:
                if rel_type == "material_family":
                    products = await self._find_material_family(
                        source_product, workspace_id, limit
                    )
                    related_products.extend(products)

                elif rel_type == "pattern_match":
                    products = await self._find_pattern_matches(
                        source_product, workspace_id, limit
                    )
                    related_products.extend(products)

                elif rel_type == "collection":
                    products = await self._find_collection_matches(
                        source_product, workspace_id, limit
                    )
                    related_products.extend(products)

                elif rel_type == "complementary":
                    products = await self._find_complementary_products(
                        source_product, workspace_id, limit
                    )
                    related_products.extend(products)

                elif rel_type == "alternative":
                    products = await self._find_alternative_products(
                        source_product, workspace_id, limit
                    )
                    related_products.extend(products)

                elif rel_type == "custom" and custom_prompt:
                    products = await self._find_custom_relationships(
                        source_product, workspace_id, custom_prompt, limit
                    )
                    related_products.extend(products)

            # Deduplicate and sort by relevance
            unique_products = self._deduplicate_products(related_products)
            sorted_products = sorted(
                unique_products,
                key=lambda x: x.get("relevance_score", 0.0),
                reverse=True
            )

            return sorted_products[:limit]

        except Exception as e:
            logger.error(f"Error finding related products: {e}", exc_info=True)
            return []

    async def _get_product(
        self, product_id: str, workspace_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get product details from database."""
        try:
            response = self.supabase.table("products").select(
                "*"
            ).eq("id", product_id).eq("workspace_id", workspace_id).single().execute()

            return response.data if response.data else None
        except Exception as e:
            logger.error(f"Error fetching product: {e}")
            return None

    async def _find_material_family(
        self, source_product: Dict, workspace_id: str, limit: int
    ) -> List[Dict[str, Any]]:
        """
        Find products with same material type.
        
        Example: Porcelain tiles matching with other porcelain tiles
        """
        try:
            metadata = source_product.get("metadata", {})
            material_type = metadata.get("material_type")

            if not material_type:
                return []

            response = self.supabase.table("products").select(
                "id, name, description, metadata"
            ).eq("workspace_id", workspace_id).neq(
                "id", source_product["id"]
            ).execute()

            related = []
            for product in response.data or []:
                prod_metadata = product.get("metadata", {})
                if prod_metadata.get("material_type") == material_type:
                    related.append({
                        "id": product["id"],
                        "name": product["name"],
                        "description": product.get("description"),
                        "metadata": prod_metadata,
                        "relationship_type": "material_family",
                        "relevance_score": 0.85,
                        "reason": f"Same material family: {material_type}"
                    })

            return related[:limit]
        except Exception as e:
            logger.error(f"Error finding material family: {e}")
            return []

    async def _find_pattern_matches(
        self, source_product: Dict, workspace_id: str, limit: int
    ) -> List[Dict[str, Any]]:
        """
        Find products with same pattern or design style.
        
        Example: Matte finish tiles matching with other matte finish tiles
        """
        try:
            metadata = source_product.get("metadata", {})
            finish = metadata.get("finish")
            colors = metadata.get("colors", [])

            if not finish:
                return []

            response = self.supabase.table("products").select(
                "id, name, description, metadata"
            ).eq("workspace_id", workspace_id).neq(
                "id", source_product["id"]
            ).execute()

            related = []
            for product in response.data or []:
                prod_metadata = product.get("metadata", {})
                if prod_metadata.get("finish") == finish:
                    # Calculate relevance based on color overlap
                    prod_colors = prod_metadata.get("colors", [])
                    color_overlap = len(set(colors) & set(prod_colors))
                    relevance = 0.75 + (0.1 * min(color_overlap, 2))

                    related.append({
                        "id": product["id"],
                        "name": product["name"],
                        "description": product.get("description"),
                        "metadata": prod_metadata,
                        "relationship_type": "pattern_match",
                        "relevance_score": relevance,
                        "reason": f"Same finish: {finish}"
                    })

            return related[:limit]
        except Exception as e:
            logger.error(f"Error finding pattern matches: {e}")
            return []

    async def _find_collection_matches(
        self, source_product: Dict, workspace_id: str, limit: int
    ) -> List[Dict[str, Any]]:
        """
        Find products from same collection or designer.
        
        Example: All products from "Harmony Collection"
        """
        try:
            metadata = source_product.get("metadata", {})
            collection = metadata.get("collection")
            designer = metadata.get("designer")
            factory = metadata.get("factory_name")

            response = self.supabase.table("products").select(
                "id, name, description, metadata"
            ).eq("workspace_id", workspace_id).neq(
                "id", source_product["id"]
            ).execute()

            related = []
            for product in response.data or []:
                prod_metadata = product.get("metadata", {})

                # Check collection match
                if collection and prod_metadata.get("collection") == collection:
                    related.append({
                        "id": product["id"],
                        "name": product["name"],
                        "description": product.get("description"),
                        "metadata": prod_metadata,
                        "relationship_type": "collection",
                        "relevance_score": 0.90,
                        "reason": f"Same collection: {collection}"
                    })

                # Check designer match
                elif designer and prod_metadata.get("designer") == designer:
                    related.append({
                        "id": product["id"],
                        "name": product["name"],
                        "description": product.get("description"),
                        "metadata": prod_metadata,
                        "relationship_type": "collection",
                        "relevance_score": 0.80,
                        "reason": f"Same designer: {designer}"
                    })

                # Check factory match
                elif factory and prod_metadata.get("factory_name") == factory:
                    related.append({
                        "id": product["id"],
                        "name": product["name"],
                        "description": product.get("description"),
                        "metadata": prod_metadata,
                        "relationship_type": "collection",
                        "relevance_score": 0.70,
                        "reason": f"Same factory: {factory}"
                    })

            return related[:limit]
        except Exception as e:
            logger.error(f"Error finding collection matches: {e}")
            return []

    async def _find_complementary_products(
        self, source_product: Dict, workspace_id: str, limit: int
    ) -> List[Dict[str, Any]]:
        """
        Find products that complement the source product.
        
        Example: Wall tiles that match floor tiles
        """
        try:
            metadata = source_product.get("metadata", {})
            category = metadata.get("category")
            material_type = metadata.get("material_type")

            if not category or not material_type:
                return []

            # Define complementary categories
            complementary_map = {
                "floor_tiles": ["wall_tiles", "trim", "border"],
                "wall_tiles": ["floor_tiles", "trim", "border"],
                "ceramic": ["porcelain", "stone"],
                "porcelain": ["ceramic", "stone"],
            }

            complementary_categories = complementary_map.get(category, [])

            response = self.supabase.table("products").select(
                "id, name, description, metadata"
            ).eq("workspace_id", workspace_id).neq(
                "id", source_product["id"]
            ).execute()

            related = []
            for product in response.data or []:
                prod_metadata = product.get("metadata", {})
                prod_category = prod_metadata.get("category")

                if prod_category in complementary_categories:
                    related.append({
                        "id": product["id"],
                        "name": product["name"],
                        "description": product.get("description"),
                        "metadata": prod_metadata,
                        "relationship_type": "complementary",
                        "relevance_score": 0.75,
                        "reason": f"Complements {category}"
                    })

            return related[:limit]
        except Exception as e:
            logger.error(f"Error finding complementary products: {e}")
            return []

    async def _find_alternative_products(
        self, source_product: Dict, workspace_id: str, limit: int
    ) -> List[Dict[str, Any]]:
        """
        Find alternative products with similar properties.
        
        Example: Different tiles with same slip resistance
        """
        try:
            metadata = source_product.get("metadata", {})
            slip_resistance = metadata.get("slip_resistance")
            fire_rating = metadata.get("fire_rating")

            response = self.supabase.table("products").select(
                "id, name, description, metadata"
            ).eq("workspace_id", workspace_id).neq(
                "id", source_product["id"]
            ).execute()

            related = []
            for product in response.data or []:
                prod_metadata = product.get("metadata", {})

                if (slip_resistance and
                    prod_metadata.get("slip_resistance") == slip_resistance):
                    related.append({
                        "id": product["id"],
                        "name": product["name"],
                        "description": product.get("description"),
                        "metadata": prod_metadata,
                        "relationship_type": "alternative",
                        "relevance_score": 0.70,
                        "reason": f"Same slip resistance: {slip_resistance}"
                    })

            return related[:limit]
        except Exception as e:
            logger.error(f"Error finding alternative products: {e}")
            return []

    async def _find_custom_relationships(
        self,
        source_product: Dict,
        workspace_id: str,
        custom_prompt: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Find related products using NLP-based custom rules.
        
        Example prompt: "Find tiles that would work in a modern minimalist kitchen"
        """
        if not self.llm:
            logger.warning("LLM service not available for custom relationships")
            return []

        try:
            # Get all products in workspace
            response = self.supabase.table("products").select(
                "id, name, description, metadata"
            ).eq("workspace_id", workspace_id).neq(
                "id", source_product["id"]
            ).execute()

            all_products = response.data or []

            # Use LLM to find related products based on custom prompt
            prompt = f"""
Given the source product and a list of candidate products, identify which products 
are related based on this criteria: {custom_prompt}

Source Product:
Name: {source_product.get('name')}
Description: {source_product.get('description')}
Metadata: {json.dumps(source_product.get('metadata', {}), indent=2)}

Candidate Products:
{json.dumps([{
    'id': p['id'],
    'name': p['name'],
    'description': p.get('description'),
    'metadata': p.get('metadata', {})
} for p in all_products[:20]], indent=2)}

Return a JSON array with the most relevant products and their relevance scores (0.0-1.0).
Format: [{{"id": "...", "relevance_score": 0.85}}]
"""

            # Call LLM to analyze relationships
            # This would use the actual LLM service
            # For now, return empty list as placeholder
            logger.info(f"Custom relationship detection with prompt: {custom_prompt}")
            return []

        except Exception as e:
            logger.error(f"Error finding custom relationships: {e}")
            return []

    def _deduplicate_products(
        self, products: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate products, keeping highest relevance score."""
        seen = {}
        for product in products:
            product_id = product["id"]
            if product_id not in seen:
                seen[product_id] = product
            else:
                # Keep the one with higher relevance score
                if product.get("relevance_score", 0) > seen[product_id].get("relevance_score", 0):
                    seen[product_id] = product

        return list(seen.values())

