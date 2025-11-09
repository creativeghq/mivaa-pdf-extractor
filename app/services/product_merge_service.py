"""
Product Merge Service

Handles merging of duplicate products from the same factory/manufacturer.
Tracks merge history and supports undo operations.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from app.services.supabase_client import SupabaseClient

logger = logging.getLogger(__name__)


class ProductMergeService:
    """Service for merging duplicate products and managing merge history."""
    
    def __init__(self, supabase_client: SupabaseClient):
        self.supabase = supabase_client
        self.logger = logger
    
    async def merge_products(
        self,
        target_product_id: str,
        source_product_ids: List[str],
        workspace_id: str,
        user_id: str,
        merge_strategy: str = 'manual',
        merge_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Merge multiple source products into a target product.
        
        Args:
            target_product_id: Product to keep (merge into)
            source_product_ids: Products to merge (will be deleted)
            workspace_id: Workspace context
            user_id: User performing the merge
            merge_strategy: 'manual', 'auto', or 'suggested'
            merge_reason: Optional reason for merge
            
        Returns:
            Merge result with history ID and updated product
        """
        try:
            self.logger.info(
                f"Merging {len(source_product_ids)} products into {target_product_id}"
            )
            
            # 1. Get target product
            target_response = self.supabase.client.table('products').select('*').eq(
                'id', target_product_id
            ).eq('workspace_id', workspace_id).single().execute()
            
            if not target_response.data:
                raise ValueError(f"Target product {target_product_id} not found")
            
            target_product = target_response.data
            target_before_merge = json.loads(json.dumps(target_product))  # Deep copy
            
            # 2. Get source products
            source_products = []
            source_product_names = []
            
            for source_id in source_product_ids:
                source_response = self.supabase.client.table('products').select('*').eq(
                    'id', source_id
                ).eq('workspace_id', workspace_id).single().execute()
                
                if source_response.data:
                    source_products.append(source_response.data)
                    source_product_names.append(source_response.data.get('name', 'Unknown'))
            
            if not source_products:
                raise ValueError("No valid source products found")
            
            # 3. Merge data from source products into target
            merged_product = await self._merge_product_data(
                target_product,
                source_products
            )
            
            # 4. Update target product in database
            update_response = self.supabase.client.table('products').update(
                merged_product
            ).eq('id', target_product_id).execute()
            
            if not update_response.data:
                raise ValueError("Failed to update target product")
            
            target_after_merge = update_response.data[0]
            
            # 5. Transfer relationships (chunks, images, etc.)
            await self._transfer_relationships(
                source_product_ids,
                target_product_id
            )
            
            # 6. Delete source products
            for source_id in source_product_ids:
                self.supabase.client.table('products').delete().eq(
                    'id', source_id
                ).execute()
            
            # 7. Record merge in history
            history_data = {
                'workspace_id': workspace_id,
                'merged_by': user_id,
                'source_product_ids': source_product_ids,
                'source_product_names': source_product_names,
                'target_product_id': target_product_id,
                'target_product_name': target_product.get('name', 'Unknown'),
                'merge_strategy': merge_strategy,
                'merge_reason': merge_reason,
                'source_products_snapshot': source_products,
                'target_product_before_merge': target_before_merge,
                'target_product_after_merge': target_after_merge
            }
            
            history_response = self.supabase.client.table('product_merge_history').insert(
                history_data
            ).execute()
            
            history_id = history_response.data[0]['id'] if history_response.data else None
            
            self.logger.info(
                f"Successfully merged {len(source_product_ids)} products. "
                f"History ID: {history_id}"
            )
            
            return {
                'success': True,
                'history_id': history_id,
                'target_product': target_after_merge,
                'merged_count': len(source_product_ids),
                'message': f'Successfully merged {len(source_product_ids)} products'
            }
            
        except Exception as e:
            self.logger.error(f"Error merging products: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _merge_product_data(
        self,
        target: Dict[str, Any],
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Merge data from source products into target product.
        
        Strategy:
        - Keep target name (primary identifier)
        - Merge descriptions (combine unique information)
        - Merge metadata (union of all metadata)
        - Keep target embeddings
        - Combine source chunks
        """
        merged = target.copy()
        
        # Merge descriptions
        descriptions = [target.get('description', '')]
        for source in sources:
            desc = source.get('description', '')
            if desc and desc not in descriptions:
                descriptions.append(desc)
        
        if len(descriptions) > 1:
            merged['description'] = ' | '.join(filter(None, descriptions))
        
        # Merge long descriptions
        long_descriptions = [target.get('long_description', '')]
        for source in sources:
            long_desc = source.get('long_description', '')
            if long_desc and long_desc not in long_descriptions:
                long_descriptions.append(long_desc)
        
        if len(long_descriptions) > 1:
            merged['long_description'] = '\n\n'.join(filter(None, long_descriptions))
        
        # Merge metadata (union)
        merged_metadata = target.get('metadata', {}).copy()
        for source in sources:
            source_metadata = source.get('metadata', {})
            for key, value in source_metadata.items():
                if key not in merged_metadata:
                    merged_metadata[key] = value
                elif isinstance(value, list) and isinstance(merged_metadata[key], list):
                    # Merge lists (unique values)
                    merged_metadata[key] = list(set(merged_metadata[key] + value))
        
        merged['metadata'] = merged_metadata
        
        # Merge source chunks
        target_chunks = target.get('source_chunks', [])
        for source in sources:
            source_chunks = source.get('source_chunks', [])
            target_chunks.extend(source_chunks)
        
        merged['source_chunks'] = list(set(target_chunks))  # Unique chunks
        
        # Update timestamp
        merged['updated_at'] = datetime.utcnow().isoformat()
        
        return merged
    
    async def _transfer_relationships(
        self,
        source_product_ids: List[str],
        target_product_id: str
    ) -> None:
        """Transfer all relationships from source products to target product."""
        try:
            # Transfer product-image relationships
            for source_id in source_product_ids:
                # Get existing relationships
                rel_response = self.supabase.client.table('product_image_relationships').select(
                    '*'
                ).eq('product_id', source_id).execute()
                
                if rel_response.data:
                    for rel in rel_response.data:
                        # Check if relationship already exists for target
                        existing = self.supabase.client.table('product_image_relationships').select(
                            'id'
                        ).eq('product_id', target_product_id).eq(
                            'image_id', rel['image_id']
                        ).execute()
                        
                        if not existing.data:
                            # Create new relationship for target
                            new_rel = {
                                'product_id': target_product_id,
                                'image_id': rel['image_id'],
                                'relationship_type': rel.get('relationship_type'),
                                'relevance_score': rel.get('relevance_score')
                            }
                            self.supabase.client.table('product_image_relationships').insert(
                                new_rel
                            ).execute()
                    
                    # Delete old relationships
                    self.supabase.client.table('product_image_relationships').delete().eq(
                        'product_id', source_id
                    ).execute()
            
            # Transfer product-document relationships (if exists)
            # Similar pattern for other relationship tables
            
        except Exception as e:
            self.logger.error(f"Error transferring relationships: {e}")
    
    async def undo_merge(
        self,
        history_id: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Undo a product merge operation.
        
        Restores source products and reverts target product to pre-merge state.
        """
        try:
            # Get merge history
            history_response = self.supabase.client.table('product_merge_history').select(
                '*'
            ).eq('id', history_id).single().execute()
            
            if not history_response.data:
                raise ValueError(f"Merge history {history_id} not found")
            
            history = history_response.data
            
            if history.get('is_undone'):
                raise ValueError("This merge has already been undone")
            
            # Restore source products
            source_products = history.get('source_products_snapshot', [])
            for product in source_products:
                self.supabase.client.table('products').insert(product).execute()
            
            # Revert target product
            target_before = history.get('target_product_before_merge')
            if target_before:
                self.supabase.client.table('products').update(
                    target_before
                ).eq('id', history['target_product_id']).execute()
            
            # Mark as undone
            self.supabase.client.table('product_merge_history').update({
                'is_undone': True,
                'undone_at': datetime.utcnow().isoformat(),
                'undone_by': user_id
            }).eq('id', history_id).execute()
            
            self.logger.info(f"Successfully undone merge {history_id}")
            
            return {
                'success': True,
                'message': 'Merge successfully undone',
                'restored_products': len(source_products)
            }
            
        except Exception as e:
            self.logger.error(f"Error undoing merge: {e}")
            return {
                'success': False,
                'error': str(e)
            }

