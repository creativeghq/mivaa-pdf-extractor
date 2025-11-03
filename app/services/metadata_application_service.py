"""
Metadata Application Service

Applies metadata to products with proper scope handling and override logic.

CRITICAL FEATURES:
1. Catalog-General Metadata: Applies to ALL products unless overridden
2. Product-Specific Metadata: Applies only to specific products, can override catalog-general
3. Processing Order: Catalog-general FIRST, then product-specific (allows overrides)

Example:
- Catalog says "Available in 15×38" → All products get dimensions: 15×38
- HARMONY says "Dimensions: 20×40" → HARMONY overrides to 20×40
- NOVA has no dimensions → Inherits 15×38 from catalog
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MetadataApplicationService:
    """
    Service for applying metadata to products with scope-aware override logic.
    """
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
        self.logger = logger
    
    async def apply_metadata_to_products(
        self,
        document_id: str,
        chunks_with_scope: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Apply metadata from chunks to products with proper scope handling.
        
        Args:
            document_id: Document ID
            chunks_with_scope: List of chunks with scope detection results
                [
                    {
                        "chunk_id": "...",
                        "content": "...",
                        "scope": "catalog_general_implicit",
                        "applies_to": "all",
                        "extracted_metadata": {"dimensions": "15×38"},
                        "is_override": false
                    },
                    ...
                ]
        
        Returns:
            {
                "products_updated": 10,
                "metadata_applied": {...},
                "overrides_detected": 2
            }
        """
        try:
            # Get all products in document
            products = await self._get_products_by_document(document_id)
            
            if not products:
                self.logger.warning(f"No products found for document {document_id}")
                return {
                    "products_updated": 0,
                    "metadata_applied": {},
                    "overrides_detected": 0
                }
            
            # Initialize product metadata tracking
            product_metadata = {p['id']: {} for p in products}
            product_overrides = {p['id']: set() for p in products}
            
            # STEP 1: Apply catalog-general metadata (explicit)
            catalog_explicit = [c for c in chunks_with_scope if c.get('scope') == 'catalog_general_explicit']
            self._apply_catalog_general(catalog_explicit, product_metadata, product_overrides)
            
            # STEP 2: Apply catalog-general metadata (implicit)
            catalog_implicit = [c for c in chunks_with_scope if c.get('scope') == 'catalog_general_implicit']
            self._apply_catalog_general(catalog_implicit, product_metadata, product_overrides)
            
            # STEP 3: Apply category-specific metadata
            category_specific = [c for c in chunks_with_scope if c.get('scope') == 'category_specific']
            self._apply_category_specific(category_specific, products, product_metadata, product_overrides)
            
            # STEP 4: Apply product-specific metadata (allows overrides)
            product_specific = [c for c in chunks_with_scope if c.get('scope') == 'product_specific']
            overrides_count = self._apply_product_specific(product_specific, products, product_metadata, product_overrides)
            
            # STEP 5: Update products in database
            updated_count = await self._update_products_in_db(products, product_metadata, product_overrides)
            
            return {
                "products_updated": updated_count,
                "metadata_applied": product_metadata,
                "overrides_detected": overrides_count
            }
        
        except Exception as e:
            self.logger.error(f"Failed to apply metadata: {e}")
            return {
                "products_updated": 0,
                "metadata_applied": {},
                "overrides_detected": 0,
                "error": str(e)
            }
    
    def _apply_catalog_general(
        self,
        chunks: List[Dict[str, Any]],
        product_metadata: Dict[str, Dict],
        product_overrides: Dict[str, set]
    ):
        """Apply catalog-general metadata to ALL products."""
        for chunk in chunks:
            extracted = chunk.get('extracted_metadata', {})
            
            for product_id in product_metadata.keys():
                # Apply metadata if not overridden
                for key, value in extracted.items():
                    if key not in product_overrides[product_id]:
                        product_metadata[product_id][key] = value
        
        self.logger.info(f"✅ Applied {len(chunks)} catalog-general chunks to all products")
    
    def _apply_category_specific(
        self,
        chunks: List[Dict[str, Any]],
        products: List[Dict],
        product_metadata: Dict[str, Dict],
        product_overrides: Dict[str, set]
    ):
        """Apply category-specific metadata to matching products."""
        for chunk in chunks:
            extracted = chunk.get('extracted_metadata', {})
            applies_to_categories = chunk.get('applies_to', [])
            
            # Find products matching category
            for product in products:
                product_id = product['id']
                product_category = product.get('metadata', {}).get('material_category', '').lower()
                
                # Check if product matches any category
                if any(cat.lower() in product_category for cat in applies_to_categories):
                    # Apply metadata if not overridden
                    for key, value in extracted.items():
                        if key not in product_overrides[product_id]:
                            product_metadata[product_id][key] = value
        
        self.logger.info(f"✅ Applied {len(chunks)} category-specific chunks")
    
    def _apply_product_specific(
        self,
        chunks: List[Dict[str, Any]],
        products: List[Dict],
        product_metadata: Dict[str, Dict],
        product_overrides: Dict[str, set]
    ) -> int:
        """Apply product-specific metadata (can override catalog-general)."""
        overrides_count = 0
        
        # Create product name → ID mapping
        product_name_to_id = {p['name'].lower(): p['id'] for p in products}
        
        for chunk in chunks:
            extracted = chunk.get('extracted_metadata', {})
            applies_to_products = chunk.get('applies_to', [])
            is_override = chunk.get('is_override', False)
            
            for product_name in applies_to_products:
                product_id = product_name_to_id.get(product_name.lower())
                
                if product_id:
                    # Apply metadata
                    for key, value in extracted.items():
                        # Check if this overrides existing catalog-general metadata
                        if key in product_metadata[product_id] and is_override:
                            self.logger.info(f"🔄 Override detected: {product_name}.{key} = {value} (was {product_metadata[product_id][key]})")
                            overrides_count += 1
                            product_overrides[product_id].add(key)
                        
                        product_metadata[product_id][key] = value
        
        self.logger.info(f"✅ Applied {len(chunks)} product-specific chunks ({overrides_count} overrides)")
        return overrides_count
    
    async def _get_products_by_document(self, document_id: str) -> List[Dict]:
        """Get all products for a document."""
        try:
            response = self.supabase.table('products').select('*').eq('document_id', document_id).execute()
            return response.data if response.data else []
        except Exception as e:
            self.logger.error(f"Failed to get products: {e}")
            return []
    
    async def _update_products_in_db(
        self,
        products: List[Dict],
        product_metadata: Dict[str, Dict],
        product_overrides: Dict[str, set]
    ) -> int:
        """Update products in database with new metadata."""
        updated_count = 0
        
        for product in products:
            product_id = product['id']
            
            try:
                # Merge new metadata with existing
                existing_metadata = product.get('metadata', {})
                new_metadata = {
                    **existing_metadata,
                    **product_metadata[product_id],
                    '_overrides': list(product_overrides[product_id])  # Track overrides
                }
                
                # Update in database
                self.supabase.table('products').update({
                    'metadata': new_metadata,
                    'updated_at': datetime.utcnow().isoformat()
                }).eq('id', product_id).execute()
                
                updated_count += 1
            
            except Exception as e:
                self.logger.error(f"Failed to update product {product_id}: {e}")
        
        self.logger.info(f"✅ Updated {updated_count} products in database")
        return updated_count


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

"""
# Example usage in PDF processing pipeline:

from app.services.dynamic_metadata_extractor import MetadataScopeDetector
from app.services.metadata_application_service import MetadataApplicationService

# Step 1: Detect scope for all chunks
scope_detector = MetadataScopeDetector()
chunks_with_scope = []

for chunk in document_chunks:
    scope_result = await scope_detector.detect_scope(
        chunk_content=chunk['content'],
        product_names=["NOVA", "HARMONY", "ESSENCE"]
    )
    
    chunks_with_scope.append({
        "chunk_id": chunk['id'],
        "content": chunk['content'],
        **scope_result
    })

# Step 2: Apply metadata to products
metadata_service = MetadataApplicationService(supabase_client)
result = await metadata_service.apply_metadata_to_products(
    document_id=document_id,
    chunks_with_scope=chunks_with_scope
)

print(f"✅ Updated {result['products_updated']} products")
print(f"🔄 Detected {result['overrides_detected']} overrides")
"""

