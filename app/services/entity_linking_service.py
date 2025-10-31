"""
Entity Linking Service

Links images, chunks, products, and metafields based on product discovery results.

Relationships implemented:
1. Image-to-Product: Links images to products based on page proximity
2. Image-to-Chunk: Links images to chunks on the same page
3. Image-to-Metafield: Links images to metafield values
4. Product-to-Metafield: Links products to metafield values
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import uuid

from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class EntityLinkingService:
    """
    Service for linking entities (images, chunks, products, metafields).
    
    Workflow:
    1. Product discovery provides image-to-product mapping
    2. This service creates database relationships
    3. Links are used for search, filtering, and product views
    """
    
    def __init__(self):
        self.logger = logger
        self.supabase = get_supabase_client()
    
    async def link_images_to_products(
        self,
        document_id: str,
        image_to_product_mapping: Dict[int, str],
        product_name_to_id: Dict[str, str]
    ) -> int:
        """
        Link images to products based on discovery mapping.
        
        Args:
            document_id: Document ID
            image_to_product_mapping: Dict mapping image index to product name
            product_name_to_id: Dict mapping product name to product ID
            
        Returns:
            Number of images linked
        """
        try:
            self.logger.info(f"🔗 Linking images to products for document {document_id}")
            
            # Get all images for this document
            response = self.supabase.client.table('document_images')\
                .select('id, page_number, metadata')\
                .eq('document_id', document_id)\
                .execute()
            
            if not response.data:
                self.logger.warning(f"⚠️ No images found for document {document_id}")
                return 0
            
            linked_count = 0
            
            for image in response.data:
                image_id = image['id']
                page_number = image['page_number']
                
                # Find product for this image
                product_name = None
                
                # Check direct mapping by image index
                metadata = image.get('metadata', {})
                image_index = metadata.get('image_index')
                
                if image_index is not None and image_index in image_to_product_mapping:
                    product_name = image_to_product_mapping[image_index]
                
                # Fallback: Find product by page proximity
                if not product_name:
                    product_name = self._find_product_by_page(
                        page_number=page_number,
                        image_to_product_mapping=image_to_product_mapping,
                        product_name_to_id=product_name_to_id
                    )
                
                if product_name and product_name in product_name_to_id:
                    product_id = product_name_to_id[product_name]
                    
                    # Update image with product_id
                    self.supabase.client.table('document_images')\
                        .update({'product_id': product_id, 'updated_at': datetime.utcnow().isoformat()})\
                        .eq('id', image_id)\
                        .execute()
                    
                    linked_count += 1
                    self.logger.debug(f"✅ Linked image {image_id} to product {product_name}")
            
            self.logger.info(f"✅ Linked {linked_count}/{len(response.data)} images to products")
            return linked_count
            
        except Exception as e:
            self.logger.error(f"❌ Failed to link images to products: {e}")
            return 0
    
    def _find_product_by_page(
        self,
        page_number: int,
        image_to_product_mapping: Dict[int, str],
        product_name_to_id: Dict[str, str]
    ) -> Optional[str]:
        """Find product by page proximity."""
        # This is a simplified implementation
        # In production, you'd use the product page ranges from discovery
        return None
    
    async def link_images_to_chunks(
        self,
        document_id: str
    ) -> int:
        """
        Link images to chunks based on page proximity.
        
        Args:
            document_id: Document ID
            
        Returns:
            Number of image-chunk links created
        """
        try:
            self.logger.info(f"🔗 Linking images to chunks for document {document_id}")
            
            # Get all images
            images_response = self.supabase.client.table('document_images')\
                .select('id, page_number')\
                .eq('document_id', document_id)\
                .execute()
            
            # Get all chunks
            chunks_response = self.supabase.client.table('document_chunks')\
                .select('id, metadata')\
                .eq('document_id', document_id)\
                .execute()
            
            if not images_response.data or not chunks_response.data:
                self.logger.warning(f"⚠️ No images or chunks found for document {document_id}")
                return 0
            
            linked_count = 0
            
            # Create page-to-chunks mapping
            page_to_chunks = {}
            for chunk in chunks_response.data:
                metadata = chunk.get('metadata', {})
                page_number = metadata.get('page_number')
                
                if page_number:
                    if page_number not in page_to_chunks:
                        page_to_chunks[page_number] = []
                    page_to_chunks[page_number].append(chunk['id'])
            
            # Link images to chunks on same page
            for image in images_response.data:
                page_number = image['page_number']
                chunk_ids = page_to_chunks.get(page_number, [])
                
                if chunk_ids:
                    # Update image metadata with related chunk IDs
                    self.supabase.client.table('document_images')\
                        .update({
                            'metadata': {
                                'related_chunk_ids': chunk_ids
                            },
                            'updated_at': datetime.utcnow().isoformat()
                        })\
                        .eq('id', image['id'])\
                        .execute()
                    
                    linked_count += 1
            
            self.logger.info(f"✅ Linked {linked_count} images to chunks")
            return linked_count
            
        except Exception as e:
            self.logger.error(f"❌ Failed to link images to chunks: {e}")
            return 0
    
    async def link_images_to_metafields(
        self,
        image_id: str,
        metafields: List[Dict[str, Any]]
    ) -> int:
        """
        Link image to metafield values.

        Args:
            image_id: Image ID
            metafields: List of metafield records

        Returns:
            Number of metafields linked
        """
        try:
            linked_count = 0

            for metafield in metafields:
                try:
                    # Determine which value column to use based on value type
                    value = metafield.get('value')
                    record = {
                        'id': str(uuid.uuid4()),
                        'image_id': image_id,
                        'field_id': metafield['field_id'],
                        'value_text': str(value) if value is not None else None,
                        'confidence_score': metafield.get('confidence', 0.9),
                        'extraction_method': 'ai_extraction',
                        'created_at': datetime.utcnow().isoformat(),
                        'updated_at': datetime.utcnow().isoformat()
                    }

                    self.supabase.client.table('image_metafield_values').insert(record).execute()
                    linked_count += 1

                except Exception as e:
                    self.logger.error(
                        f"❌ Failed to link metafield {metafield.get('field_name', 'unknown')} to image {image_id}: {e}"
                    )

            return linked_count

        except Exception as e:
            self.logger.error(f"❌ Failed to link metafields to image: {e}")
            return 0
    
    async def link_products_to_metafields(
        self,
        product_id: str,
        metafields: List[Dict[str, Any]]
    ) -> int:
        """
        Link product to metafield values.

        Args:
            product_id: Product ID
            metafields: List of metafield records

        Returns:
            Number of metafields linked
        """
        try:
            linked_count = 0

            for metafield in metafields:
                try:
                    # Determine which value column to use based on value type
                    value = metafield.get('value')
                    record = {
                        'id': str(uuid.uuid4()),
                        'product_id': product_id,
                        'field_id': metafield['field_id'],
                        'value_text': str(value) if value is not None else None,
                        'confidence_score': metafield.get('confidence', 0.9),
                        'extraction_method': 'ai_extraction',
                        'created_at': datetime.utcnow().isoformat(),
                        'updated_at': datetime.utcnow().isoformat()
                    }

                    self.supabase.client.table('product_metafield_values').insert(record).execute()
                    linked_count += 1

                except Exception as e:
                    self.logger.error(
                        f"❌ Failed to link metafield {metafield.get('field_name', 'unknown')} to product {product_id}: {e}"
                    )

            self.logger.info(f"✅ Linked {linked_count}/{len(metafields)} metafields to product {product_id}")
            return linked_count

        except Exception as e:
            self.logger.error(f"❌ Failed to link metafields to product: {e}")
            return 0
    
    async def link_all_entities(
        self,
        document_id: str,
        catalog: Any  # ProductCatalog object from discovery
    ) -> Dict[str, int]:
        """
        Link all entities for a document.

        Args:
            document_id: Document ID
            catalog: ProductCatalog from discovery service

        Returns:
            Statistics of links created
        """
        try:
            self.logger.info(f"🔗 Linking all entities for document {document_id}")

            stats = {
                'image_product_links': 0,
                'image_chunk_links': 0,
                'image_metafield_links': 0,
                'product_metafield_links': 0,
                'metafield_links': 0
            }

            # Get all products from database to build name-to-id mapping
            products_response = self.supabase.client.table('products')\
                .select('id, name')\
                .eq('source_document_id', document_id)\
                .execute()

            product_name_to_id = {p['name']: p['id'] for p in products_response.data}

            # 1. Link images to products based on discovery image_to_product_mapping
            image_to_product_mapping = {}
            if hasattr(catalog, 'image_to_product_mapping'):
                image_to_product_mapping = catalog.image_to_product_mapping
            else:
                # Build from products
                for product in catalog.products:
                    if hasattr(product, 'image_indices'):
                        for image_index in product.image_indices:
                            image_to_product_mapping[image_index] = product.name

            stats['image_product_links'] = await self.link_images_to_products(
                document_id=document_id,
                image_to_product_mapping=image_to_product_mapping,
                product_name_to_id=product_name_to_id
            )

            # 2. Link images to chunks (page proximity)
            stats['image_chunk_links'] = await self.link_images_to_chunks(
                document_id=document_id
            )

            # 3. Link products to metafields
            # Get metafield values from product_metafield_values table
            metafields_response = self.supabase.client.table('product_metafield_values')\
                .select('product_id, field_id, value')\
                .in_('product_id', list(product_name_to_id.values()))\
                .execute()

            # Count unique product-metafield links
            stats['product_metafield_links'] = len(metafields_response.data)
            stats['metafield_links'] = stats['product_metafield_links'] + stats['image_metafield_links']

            self.logger.info(f"✅ Entity linking complete:")
            self.logger.info(f"   - Image → Product: {stats['image_product_links']}")
            self.logger.info(f"   - Image → Chunk: {stats['image_chunk_links']}")
            self.logger.info(f"   - Product → Metafield: {stats['product_metafield_links']}")
            self.logger.info(f"   - Total metafield links: {stats['metafield_links']}")

            return stats

        except Exception as e:
            self.logger.error(f"❌ Failed to link all entities: {e}")
            return {
                'image_product_links': 0,
                'image_chunk_links': 0,
                'image_metafield_links': 0,
                'product_metafield_links': 0,
                'metafield_links': 0
            }

