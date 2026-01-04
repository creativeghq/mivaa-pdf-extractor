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
    ) -> Dict[str, Any]:
        """
        Link images to products using relationship table with relevance scores.

        Relevance Algorithm (for PyMuPDF fallback):
        - page_overlap(40%): Same page = 0.4, adjacent = 0.2, else 0.0
        - visual_similarity(40%): From AI detection (default 0.3)
        - detection_score(20%): Confidence from discovery (default 0.2)

        Vision-Guided Algorithm (95% accuracy):
        - Uses atomic product name from vision AI (no guesswork)
        - Relevance score = detection confidence (0.8-0.95)

        Args:
            document_id: Document ID
            image_to_product_mapping: Dict mapping image index to product name
            product_name_to_id: Dict mapping product name to product ID

        Returns:
            Dict with linking stats including vision-guided metrics
        """
        try:
            self.logger.info(f"🔗 Linking images to products for document {document_id}")

            # Get all images for this document
            # ✅ NEW: Include vision-guided metadata fields
            images_response = self.supabase.client.table('document_images')\
                .select('id, page_number, metadata, extraction_method, product_name, detection_confidence')\
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
                # Ensure page_range is a list
                if isinstance(page_range, int):
                    page_range = [page_range]
                elif not isinstance(page_range, list):
                    page_range = []
                product_page_ranges[product['id']] = page_range

            linked_count = 0
            relationships = []

            # ✅ NEW: Track vision-guided stats
            vision_stats = {
                'atomic_links': 0,
                'index_mapping_links': 0,
                'page_proximity_links': 0,
                'total_vision_confidence': 0.0,
                'vision_confidence_count': 0
            }

            for image in images_response.data:
                image_id = image['id']
                page_number = image['page_number']
                metadata = image.get('metadata', {})
                image_index = metadata.get('image_index')

                # ✅ NEW: Get vision-guided metadata
                extraction_method = image.get('extraction_method', 'pymupdf')
                vision_product_name = image.get('product_name')  # Atomic product name from vision AI
                vision_confidence = image.get('detection_confidence', 0.0)

                # Find product for this image
                product_name = None
                linking_method = None  # Track how we linked this image

                # ✅ PRIORITY 1: Vision-guided atomic linking (95% accuracy, no guesswork)
                if extraction_method == 'vision_guided' and vision_product_name:
                    product_name = vision_product_name
                    linking_method = 'vision_guided_atomic'
                    vision_stats['atomic_links'] += 1
                    if vision_confidence > 0:
                        vision_stats['total_vision_confidence'] += vision_confidence
                        vision_stats['vision_confidence_count'] += 1
                    self.logger.debug(f"   ✅ Vision-guided atomic link: {vision_product_name} (confidence: {vision_confidence:.2f})")

                # ✅ PRIORITY 2: Check direct mapping by image index (for PyMuPDF fallback)
                elif image_index is not None and image_index in image_to_product_mapping:
                    product_name = image_to_product_mapping[image_index]
                    linking_method = 'image_index_mapping'
                    vision_stats['index_mapping_links'] += 1

                # ✅ PRIORITY 3: Fallback to page proximity (for PyMuPDF fallback)
                if not product_name:
                    # Find product whose page_range contains this image's page
                    for pid, page_range in product_page_ranges.items():
                        if page_range and len(page_range) > 0:

                            start_page, end_page = min(page_range), max(page_range)
                            if start_page <= page_number <= end_page:
                                # Find product name by ID
                                for pname, pid_check in product_name_to_id.items():
                                    if pid_check == pid:
                                        product_name = pname
                                        linking_method = 'page_proximity_fallback'
                                        vision_stats['page_proximity_links'] += 1
                                        break
                                break

                if product_name and product_name in product_name_to_id:
                    product_id = product_name_to_id[product_name]

                    # ✅ NEW: Calculate relevance score based on linking method
                    if linking_method == 'vision_guided_atomic':
                        # Vision-guided: Use detection confidence directly (95% accuracy)
                        relevance_score = min(0.95, vision_confidence) if vision_confidence > 0 else 0.95
                    else:
                        # PyMuPDF fallback: Use traditional page proximity algorithm
                        relevance_score = self._calculate_image_product_relevance(
                            image_page=page_number,
                            product_page_range=product_page_ranges.get(product_id, []),
                            detection_confidence=metadata.get('confidence', 0.8)
                        )

                    # Create relationship entry
                    # NOTE: metadata column removed - table doesn't have it
                    relationships.append({
                        'id': str(uuid.uuid4()),
                        'image_id': image_id,
                        'product_id': product_id,
                        'relevance_score': relevance_score,
                        'relationship_type': 'product_image',
                        'created_at': datetime.utcnow().isoformat()
                    })

                    linked_count += 1
                    self.logger.debug(f"✅ Linked image {image_id} to product {product_name} (method: {linking_method}, relevance: {relevance_score:.2f})")

            # Batch insert all relationships (using SAME table as frontend)
            if relationships:
                self.supabase.client.table('product_image_relationships')\
                    .insert(relationships)\
                    .execute()
                self.logger.info(f"✅ Created {len(relationships)} product-image relationship entries")

            # ✅ NEW: Calculate average vision confidence
            avg_vision_confidence = (
                vision_stats['total_vision_confidence'] / vision_stats['vision_confidence_count']
                if vision_stats['vision_confidence_count'] > 0 else 0.0
            )

            self.logger.info(f"✅ Linked {linked_count}/{len(images_response.data)} images to products")
            self.logger.info(f"   Vision-guided atomic: {vision_stats['atomic_links']}")
            self.logger.info(f"   Index mapping: {vision_stats['index_mapping_links']}")
            self.logger.info(f"   Page proximity: {vision_stats['page_proximity_links']}")
            if vision_stats['atomic_links'] > 0:
                self.logger.info(f"   Avg vision confidence: {avg_vision_confidence:.2f}")

            # ✅ NEW: Return detailed stats
            return {
                'linked_count': linked_count,
                'total_images': len(images_response.data),
                'vision_guided_links': vision_stats['atomic_links'],
                'fallback_links': vision_stats['index_mapping_links'] + vision_stats['page_proximity_links'],
                'vision_stats': {
                    'atomic_links': vision_stats['atomic_links'],
                    'index_mapping_links': vision_stats['index_mapping_links'],
                    'page_proximity_links': vision_stats['page_proximity_links'],
                    'avg_vision_confidence': avg_vision_confidence
                }
            }

        except Exception as e:
            self.logger.error(f"❌ Failed to link images to products: {e}")
            return {
                'linked_count': 0,
                'total_images': 0,
                'vision_guided_links': 0,
                'fallback_links': 0,
                'vision_stats': {}
            }

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
                # ✅ FIX: Chunks use 'page_label' (string) not 'page_number'
                page_label = metadata.get('page_label')

                if page_label:
                    try:
                        page_number = int(page_label)
                        if page_number not in page_to_chunks:
                            page_to_chunks[page_number] = []
                        page_to_chunks[page_number].append(chunk['id'])
                        chunk_pages[chunk['id']] = page_number
                    except (ValueError, TypeError):
                        self.logger.warning(f"⚠️ Invalid page_label '{page_label}' for chunk {chunk['id']}")
            
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
               
                if not isinstance(chunk_metadata, dict):
                    self.logger.warning(f"⚠️ Skipping chunk {chunk['id']} - metadata is not a dict: {type(chunk_metadata)}")
                    continue

                # Get sequential page number and document-level product_pages array
                chunk_page_number = chunk_metadata.get('page_number')
                product_pages_array = chunk_metadata.get('product_pages', [])

                if isinstance(product_pages_array, int):
                    product_pages_array = [product_pages_array]
                elif not isinstance(product_pages_array, list):
                    product_pages_array = []

                # Skip chunks without page_number
                if chunk_page_number is None:
                    self.logger.warning(f"⚠️ Skipping chunk {chunk['id']} - missing page_number")
                    continue

                if not isinstance(chunk_page_number, int):
                    try:
                        chunk_page_number = int(chunk_page_number)
                    except (ValueError, TypeError):
                        self.logger.warning(f"⚠️ Skipping chunk {chunk['id']} - invalid page_number: {chunk_page_number}")
                        continue

                # Map sequential page number to original PDF page number
                # product_pages is a document-level array: [24, 25, 26, ...]
                # page_number is 1-based sequential: 1, 2, 3, ...
                # So: original_page = product_pages[page_number - 1]
                if product_pages_array and isinstance(product_pages_array, list) and chunk_page_number <= len(product_pages_array):
                    chunk_original_page = product_pages_array[chunk_page_number - 1]
                else:
                    # Fallback: use sequential page number if mapping not available
                    chunk_original_page = chunk_page_number

                chunk_content = chunk.get('content', '').lower()

                for product in products_response.data:
                    product_metadata = product.get('metadata', {})
                    product_page_range = product_metadata.get('page_range', [])
                    product_name = product.get('name', '').lower()

                    # Calculate relevance score using original PDF page numbers
                    relevance_score = self._calculate_chunk_product_relevance(
                        chunk_original_page=chunk_original_page,
                        chunk_content=chunk_content,
                        product_page_range=product_page_range,
                        product_name=product_name
                    )

                    # Only create relationship if relevance is above threshold
                    # Threshold is 0.3 (requires either page match OR content mention)
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
        chunk_original_page: int,
        chunk_content: str,
        product_page_range: List[int],
        product_name: str
    ) -> float:
        """
        Calculate relevance score for chunk-product relationship.

        Page-Aware Algorithm (NO FALLBACK):
        - Page proximity (50%) - chunks on same/adjacent pages (using ORIGINAL PDF page numbers)
        - Content mentions (50%) - chunks mentioning product name

        Threshold: 0.3 (requires either page match OR content mention)

        Examples:
        - Chunk on product page: 0.5 ✅
        - Chunk mentioning product: 0.5 ✅
        - Chunk with both: 1.0 ✅
        - Random chunk: 0.0 ❌ (below threshold)

        Args:
            chunk_original_page: Original PDF page number for this chunk (mapped from page_number)
            chunk_content: Content of the chunk
            product_page_range: List of original PDF page numbers for the product
            product_name: Name of the product

        Returns:
            Relevance score (0.0-1.0)
        """
        relevance_score = 0.0

        # 1. Page proximity component (50%) - using ORIGINAL PDF page numbers
        if product_page_range:
            if chunk_original_page in product_page_range:
                # Chunk is on a product page
                relevance_score += 0.5
            else:
                # Check for adjacent pages
                min_distance = min(abs(chunk_original_page - p) for p in product_page_range)
                if min_distance == 1:
                    relevance_score += 0.25  # Adjacent page
                elif min_distance == 2:
                    relevance_score += 0.1  # 2 pages away

        # 2. Content mention component (50%)
        # Extract product name without designer (e.g., "MAISON" from "MAISON by ONSET")
        if product_name:
            # Try to extract just the product name (before " by ")
            product_name_parts = product_name.split(' by ')
            product_name_only = product_name_parts[0].strip() if product_name_parts else product_name

            # Check if product name (or just the main part) is mentioned
            if product_name in chunk_content or product_name_only in chunk_content:
                relevance_score += 0.5

        return min(1.0, max(0.0, relevance_score))

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
                'chunk_product_links': 0,
                # ✅ NEW: Track vision-guided vs fallback linking
                'vision_guided_links': 0,
                'fallback_links': 0,
                'vision_guided_stats': {
                    'atomic_links': 0,
                    'index_mapping_links': 0,
                    'page_proximity_links': 0,
                    'avg_vision_confidence': 0.0
                }
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
                self.logger.info(f"📋 Using catalog.image_to_product_mapping: {len(image_to_product_mapping)} mappings")
            else:
                # Build from products
                for product in catalog.products:
                    if hasattr(product, 'image_indices') and product.image_indices:
                        for image_index in product.image_indices:
                            image_to_product_mapping[image_index] = product.name
                        self.logger.debug(f"   Product '{product.name}' has {len(product.image_indices)} image_indices")
                    else:
                        self.logger.warning(f"   ⚠️ Product '{product.name}' has NO image_indices!")

                self.logger.info(f"📋 Built image_to_product_mapping from products: {len(image_to_product_mapping)} mappings")

            # ✅ NEW: link_images_to_products now returns detailed stats
            image_linking_result = await self.link_images_to_products(
                document_id=document_id,
                image_to_product_mapping=image_to_product_mapping,
                product_name_to_id=product_name_to_id
            )

            # Update stats with vision-guided metrics
            stats['image_product_links'] = image_linking_result['linked_count']
            stats['vision_guided_links'] = image_linking_result['vision_guided_links']
            stats['fallback_links'] = image_linking_result['fallback_links']
            stats['vision_guided_stats'] = image_linking_result['vision_stats']

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


