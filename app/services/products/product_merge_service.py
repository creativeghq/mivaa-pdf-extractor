"""
Product Merge Service

Handles merging of duplicate products from the same factory/manufacturer.
Tracks merge history and supports undo operations.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from app.services.core.supabase_client import SupabaseClient

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
            
            # 3. Worked out first: a failure here must leave every record where it was.
            merged_product = await self._merge_product_data(
                target_product,
                source_products
            )

            # 4. Every record pointing at a source moves to the target in one transaction, or nothing does.
            moved = self.supabase.client.rpc('merge_products_into', {
                'p_target': target_product_id, 'p_sources': source_product_ids,
            }).execute().data or {}

            # 5. Update target product in database
            update_response = self.supabase.client.table('products').update(
                merged_product
            ).eq('id', target_product_id).execute()

            if not update_response.data:
                raise ValueError("Failed to update target product")

            target_after_merge = update_response.data[0]

            # 6. The survivor's text changed, so its vector did: unless its catalogue is waiting on its embed run.
            await self._reembed(target_after_merge)

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
                'moved': moved,
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
        """Only the fields a merge changes: the survivor's own values win and a source fills what it lacks."""
        merged: Dict[str, Any] = {}
        for field in ('description', 'long_description'):
            filled = target.get(field) or next((p.get(field) for p in sources if p.get(field)), None)
            if filled and filled != target.get(field):
                merged[field] = filled
        metadata = dict(target.get('metadata') or {})
        attributes = dict(target.get('attributes') or {})
        for source in sources:
            self._union_into(metadata, source.get('metadata') or {})
            self._union_into(attributes, source.get('attributes') or {})
        metadata['merged_from'] = (metadata.get('merged_from') or []) + [
            {'id': p['id'], 'name': p.get('name')} for p in sources]
        merged['metadata'] = metadata
        merged['attributes'] = attributes
        merged['updated_at'] = datetime.utcnow().isoformat()
        return merged

    @staticmethod
    def _union_into(into: Dict[str, Any], other: Dict[str, Any]) -> None:
        for key, value in other.items():
            if key not in into or into[key] in (None, '', [], {}):
                into[key] = value
            elif isinstance(value, list) and isinstance(into[key], list):
                seen = {json.dumps(v, sort_keys=True, default=str) for v in into[key]}
                into[key] = into[key] + [v for v in value if json.dumps(v, sort_keys=True, default=str) not in seen]

    async def _reembed(self, product: Dict[str, Any]) -> None:
        from app.services.embeddings.embedding_deferral import document_deferred
        from app.services.embeddings.text_embedding_backfill import backfill_product_text_embeddings

        if document_deferred(self.supabase, product.get('source_document_id')):
            return
        self.supabase.client.table('products').update({'text_embedding_1024': None}).eq('id', product['id']).execute()
        await backfill_product_text_embeddings(max_products=1, product_ids=[product['id']])

    async def drop_product(self, product_id: str, workspace_id: str) -> Dict[str, Any]:
        """Remove a product discovery should not have made; its text stays with the document."""
        found = self.supabase.client.table('products').select('id').eq('id', product_id).eq('workspace_id', workspace_id).execute().data
        if not found:
            raise ValueError('Product not found')
        return self.supabase.client.rpc('drop_catalog_product', {'p_product': product_id}).execute().data or {}

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


