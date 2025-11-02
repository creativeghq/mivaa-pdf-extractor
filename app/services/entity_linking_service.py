"""
Entity Linking Service

Links images, chunks, and products using the SAME relationship tables as frontend.

IMPORTANT: This service uses the SAME tables as entityRelationshipService.ts:
- chunk_product_relationships (chunk_id, product_id, relevance_score)
- chunk_image_relationships (chunk_id, image_id, relevance_score)
- product_image_relationships (product_id, image_id, relevance_score)

Relationships implemented:
1. Product → Image: Links products to images based on page proximity and visual similarity
2. Chunk → Image: Links chunks to images on the same page with spatial proximity
3. Chunk → Product: Links chunks to products based on page proximity and content similarity

All relationships are stored with relevance scores (0.0-1.0).
"""

import logging
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
import uuid

from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class EntityLinkingService:
    """
    Service for linking entities (images, chunks, products) using relationship tables.

    CRITICAL: Uses the SAME tables as frontend entityRelationshipService.ts to avoid duplicates.

    Tables used:
    - chunk_product_relationships
    - chunk_image_relationships
    - product_image_relationships

    Relevance Score Algorithms:
    - Product → Image: page_overlap(40%) + visual_similarity(40%) + detection_score(20%)
    - Chunk → Image: same_page(50%) + visual_text_similarity(30%) + spatial_proximity(20%)
    - Chunk → Product: page_proximity(40%) + embedding_similarity(30%) + mention_score(30%)
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
        Link images to products using relationship table with relevance scores.

        Relevance Algorithm:
        - page_overlap(40%): Same page = 0.4, adjacent = 0.2, else 0.0
        - visual_similarity(40%): From AI detection (default 0.3)
        - detection_score(20%): Confidence from discovery (default 0.2)

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
            images_response = self.supabase.client.table('document_images')\
                .select('id, page_number, metadata')\
                .eq('document_id', document_id)\
                .execute()

            if not images_response.data:
                self.logger.warning(f"⚠️ No images found for document {document_id}")
                return 0

            # Get all products with page ranges
            products_response = self.supabase.client.table('products')\
                .select('id, name, metadata')\
                .eq('source_document_id', document_id)\
                .execute()

            product_page_ranges = {}
            for product in products_response.data:
                metadata = product.get('metadata', {})
                page_range = metadata.get('page_range', [])
                product_page_ranges[product['id']] = page_range

            linked_count = 0
            relationships = []

            for image in images_response.data:
                image_id = image['id']
                page_number = image['page_number']
                metadata = image.get('metadata', {})
                image_index = metadata.get('image_index')

                # Find product for this image
                product_name = None

                # Check direct mapping by image index
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

                    # Calculate relevance score
                    relevance_score = self._calculate_image_product_relevance(
                        image_page=page_number,
                        product_page_range=product_page_ranges.get(product_id, []),
                        detection_confidence=metadata.get('confidence', 0.8)
                    )

                    # Create relationship entry
                    relationships.append({
                        'id': str(uuid.uuid4()),
                        'image_id': image_id,
                        'product_id': product_id,
                        'relevance_score': relevance_score,
                        'relationship_type': 'product_image',
                        'created_at': datetime.utcnow().isoformat()
                    })

                    # Also update image with product_id for backward compatibility
                    self.supabase.client.table('document_images')\
                        .update({'product_id': product_id, 'updated_at': datetime.utcnow().isoformat()})\
                        .eq('id', image_id)\
                        .execute()

                    linked_count += 1
                    self.logger.debug(f"✅ Linked image {image_id} to product {product_name} (relevance: {relevance_score:.2f})")

            # Batch insert all relationships (using SAME table as frontend)
            if relationships:
                self.supabase.client.table('product_image_relationships')\
                    .insert(relationships)\
                    .execute()
                self.logger.info(f"✅ Created {len(relationships)} product-image relationship entries")

            self.logger.info(f"✅ Linked {linked_count}/{len(images_response.data)} images to products")
            return linked_count

        except Exception as e:
            self.logger.error(f"❌ Failed to link images to products: {e}")
            return 0

    def _calculate_image_product_relevance(
        self,
        image_page: int,
        product_page_range: List[int],
        detection_confidence: float = 0.8
    ) -> float:
        """
        Calculate relevance score for image-product relationship.

        Algorithm: page_overlap(40%) + visual_similarity(40%) + detection_score(20%)

        Args:
            image_page: Page number of the image
            product_page_range: List of page numbers for the product
            detection_confidence: Confidence from AI detection

        Returns:
            Relevance score (0.0-1.0)
        """
        # Page overlap component (40%)
        page_overlap_score = 0.0
        if product_page_range and image_page in product_page_range:
            page_overlap_score = 0.4  # Same page
        elif product_page_range:
            # Check adjacent pages
            min_distance = min(abs(image_page - p) for p in product_page_range)
            if min_distance == 1:
                page_overlap_score = 0.2  # Adjacent page
            elif min_distance == 2:
                page_overlap_score = 0.1  # 2 pages away

        # Visual similarity component (40%) - use detection confidence as proxy
        visual_similarity_score = detection_confidence * 0.4

        # Detection score component (20%)
        detection_score = detection_confidence * 0.2

        total_score = page_overlap_score + visual_similarity_score + detection_score
        return min(1.0, max(0.0, total_score))

    def _calculate_image_chunk_relevance(
        self,
        image_page: int,
        chunk_page: int
    ) -> float:
        """
        Calculate relevance score for image-chunk relationship.

        Algorithm: same_page(50%) + visual_text_similarity(30%) + spatial_proximity(20%)

        Args:
            image_page: Page number of the image
            chunk_page: Page number of the chunk

        Returns:
            Relevance score (0.0-1.0)
        """
        # Same page component (50%)
        same_page_score = 0.5 if image_page == chunk_page else 0.0

        # Visual-text similarity (30%) - default medium relevance
        visual_text_score = 0.15 if image_page == chunk_page else 0.0

        # Spatial proximity (20%) - default medium relevance
        spatial_score = 0.1 if image_page == chunk_page else 0.0

        total_score = same_page_score + visual_text_score + spatial_score
        return min(1.0, max(0.0, total_score))

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
        Link images to chunks using relationship table with relevance scores.

        Relevance Algorithm:
        - same_page(50%): Images and chunks on same page
        - visual_text_similarity(30%): Default medium relevance
        - spatial_proximity(20%): Default medium relevance

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

            relationships = []

            # Create page-to-chunks mapping
            page_to_chunks = {}
            chunk_pages = {}
            for chunk in chunks_response.data:
                metadata = chunk.get('metadata', {})
                page_number = metadata.get('page_number')

                if page_number:
                    if page_number not in page_to_chunks:
                        page_to_chunks[page_number] = []
                    page_to_chunks[page_number].append(chunk['id'])
                    chunk_pages[chunk['id']] = page_number
            
            # Link images to chunks on same page
            for image in images_response.data:
                image_page = image['page_number']
                chunk_ids = page_to_chunks.get(image_page, [])

                for chunk_id in chunk_ids:
                    chunk_page = chunk_pages.get(chunk_id, image_page)

                    # Calculate relevance score
                    relevance_score = self._calculate_image_chunk_relevance(
                        image_page=image_page,
                        chunk_page=chunk_page
                    )

                    # Create relationship entry (matching frontend schema)
                    relationships.append({
                        'id': str(uuid.uuid4()),
                        'chunk_id': chunk_id,
                        'image_id': image['id'],
                        'relevance_score': relevance_score,
                        'relationship_type': 'page_proximity',
                        'created_at': datetime.utcnow().isoformat()
                    })

            # Batch insert all relationships (using SAME table as frontend)
            if relationships:
                self.supabase.client.table('chunk_image_relationships')\
                    .insert(relationships)\
                    .execute()
                self.logger.info(f"✅ Created {len(relationships)} chunk-image relationship entries")

            return len(relationships)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to link images to chunks: {e}")
            return 0

    async def link_chunks_to_products(
        self,
        document_id: str
    ) -> int:
        """
        Link chunks to products using relationship table with relevance scores.

        Relevance Algorithm:
        - page_proximity(40%): Same page = 0.4, adjacent = 0.2, else 0.0
        - embedding_similarity(30%): Default medium relevance (0.15)
        - mention_score(30%): Product name mentioned in chunk (0.3 if yes, 0.0 if no)

        Args:
            document_id: Document ID

        Returns:
            Number of chunk-product links created
        """
        try:
            self.logger.info(f"🔗 Linking chunks to products for document {document_id}")

            # Get all chunks
            chunks_response = self.supabase.client.table('document_chunks')\
                .select('id, content, metadata')\
                .eq('document_id', document_id)\
                .execute()

            # Get all products with page ranges
            products_response = self.supabase.client.table('products')\
                .select('id, name, metadata')\
                .eq('source_document_id', document_id)\
                .execute()

            if not chunks_response.data or not products_response.data:
                self.logger.warning(f"⚠️ No chunks or products found for document {document_id}")
                return 0

            relationships = []

            for chunk in chunks_response.data:
                chunk_metadata = chunk.get('metadata', {})
                chunk_page = chunk_metadata.get('page_number')
                chunk_content = chunk.get('content', '').lower()

                for product in products_response.data:
                    product_metadata = product.get('metadata', {})
                    product_page_range = product_metadata.get('page_range', [])
                    product_name = product.get('name', '').lower()

                    # Calculate relevance score
                    relevance_score = self._calculate_chunk_product_relevance(
                        chunk_page=chunk_page,
                        chunk_content=chunk_content,
                        product_page_range=product_page_range,
                        product_name=product_name
                    )

                    # Only create relationship if relevance is above threshold
                    if relevance_score >= 0.3:
                        relationships.append({
                            'id': str(uuid.uuid4()),
                            'chunk_id': chunk['id'],
                            'product_id': product['id'],
                            'relationship_type': 'source',
                            'relevance_score': relevance_score,
                            'created_at': datetime.utcnow().isoformat()
                        })

            # Batch insert all relationships
            if relationships:
                self.supabase.client.table('chunk_product_relationships')\
                    .insert(relationships)\
                    .execute()
                self.logger.info(f"✅ Created {len(relationships)} chunk-product relationship entries")

            return len(relationships)

        except Exception as e:
            self.logger.error(f"❌ Failed to link chunks to products: {e}")
            return 0

    def _calculate_chunk_product_relevance(
        self,
        chunk_page: Optional[int],
        chunk_content: str,
        product_page_range: List[int],
        product_name: str
    ) -> float:
        """
        Calculate relevance score for chunk-product relationship.

        Algorithm: page_proximity(40%) + embedding_similarity(30%) + mention_score(30%)

        Args:
            chunk_page: Page number of the chunk
            chunk_content: Content of the chunk
            product_page_range: List of page numbers for the product
            product_name: Name of the product

        Returns:
            Relevance score (0.0-1.0)
        """
        # Page proximity component (40%)
        page_proximity_score = 0.0
        if chunk_page and product_page_range:
            if chunk_page in product_page_range:
                page_proximity_score = 0.4  # Same page
            else:
                # Check adjacent pages
                min_distance = min(abs(chunk_page - p) for p in product_page_range)
                if min_distance == 1:
                    page_proximity_score = 0.2  # Adjacent page
                elif min_distance == 2:
                    page_proximity_score = 0.1  # 2 pages away

        # Embedding similarity component (30%) - default medium relevance
        embedding_similarity_score = 0.15

        # Mention score component (30%) - check if product name is mentioned
        mention_score = 0.0
        if product_name and product_name in chunk_content:
            mention_score = 0.3

        total_score = page_proximity_score + embedding_similarity_score + mention_score
        return min(1.0, max(0.0, total_score))

    async def link_all_entities(
        self,
        document_id: str,
        catalog: Any  # ProductCatalog object from discovery
    ) -> Dict[str, int]:
        """
        Link all entities for a document using relationship tables.

        Creates three types of relationships:
        1. Image → Product (with relevance scores)
        2. Image → Chunk (with relevance scores)
        3. Chunk → Product (with relevance scores)

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
                'chunk_product_links': 0
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

            # 3. Link chunks to products (page proximity + content similarity)
            stats['chunk_product_links'] = await self.link_chunks_to_products(
                document_id=document_id
            )

            self.logger.info(f"✅ Entity linking complete:")
            self.logger.info(f"   - Image → Product: {stats['image_product_links']}")
            self.logger.info(f"   - Image → Chunk: {stats['image_chunk_links']}")
            self.logger.info(f"   - Chunk → Product: {stats['chunk_product_links']}")

            return stats

        except Exception as e:
            self.logger.error(f"❌ Failed to link all entities: {e}")
            return {
                'image_product_links': 0,
                'image_chunk_links': 0,
                'chunk_product_links': 0
            }

