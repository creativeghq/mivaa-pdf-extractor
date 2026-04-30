"""
Relevancy Service - Handles relationships between chunks, images, and products.

This service creates and manages relevancy relationships:
1. Chunk-to-image relationships
2. Product-to-image relationships
3. Product-to-chunk relationships
4. Calculate relevancy scores using embeddings
"""

import logging
from typing import List, Dict, Any, Optional
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class RelevancyService:
    """Service for managing entity relationships and relevancy scores."""

    def __init__(self):
        self.supabase_client = get_supabase_client()


    # chunk_image_relationships are produced by
    # `entity_linking_service.link_images_to_chunks` (page proximity), not here.

    async def create_product_image_relationships(
        self,
        document_id: str,
        product_ids: List[str]
    ) -> int:
        """
        Create product-to-image associations based on page numbers and metadata.

        Args:
            document_id: Document ID
            product_ids: List of product IDs

        Returns:
            Number of relationships created
        """
        logger.info(f"🔗 Creating product-image relationships for {len(product_ids)} products...")

        relationships_created = 0
        products_processed = 0
        products_without_pages = 0

        for product_id in product_ids:
            try:
                # Get product metadata (page ranges)
                product_result = self.supabase_client.client.table('products')\
                    .select('id, name, metadata')\
                    .eq('id', product_id)\
                    .execute()

                if not product_result.data or len(product_result.data) == 0:
                    logger.warning(f"   ⚠️ Product {product_id} not found in database")
                    continue

                product = product_result.data[0]
                product_name = product.get('name', 'Unknown')
                metadata = product.get('metadata', {})
                page_range = metadata.get('page_range', [])

                logger.debug(f"   Processing product: {product_name} (ID: {product_id})")
                logger.debug(f"   Page range: {page_range}")

                if not page_range:
                    logger.warning(f"   ⚠️ Product {product_name} has no page_range in metadata")
                    products_without_pages += 1

                    # FALLBACK: Try to link ALL images from this document to this product
                    # This ensures products get linked even without page range metadata
                    logger.info(f"   🔄 Fallback: Linking all document images to product {product_name}")
                    images_result = self.supabase_client.client.table('document_images')\
                        .select('id, page_number')\
                        .eq('document_id', document_id)\
                        .execute()

                    images = images_result.data if images_result.data else []
                    logger.info(f"   Found {len(images)} images to link (fallback mode)")

                    for image in images:
                        try:
                            relationship = {
                                'product_id': product_id,
                                'image_id': image['id'],
                                'spatial_score': 0.0,
                                'caption_score': 0.0,
                                'clip_score': 0.0,
                                'overall_score': 0.7,  # Lower score for fallback matching
                                'confidence': 0.7,
                                'reasoning': 'document_association',  # replaces relationship_type
                                'metadata': {'fallback': True}
                            }

                            self.supabase_client.client.table('image_product_associations')\
                                .insert(relationship)\
                                .execute()

                            relationships_created += 1
                            logger.debug(f"   ✅ Linked image {image['id']} to product {product_name} (fallback)")

                        except Exception as e:
                            logger.warning(f"   ⚠️ Failed to create fallback relationship: {e}")

                    products_processed += 1
                    continue

                # page_range is a flat list of page numbers, e.g. [12, 13, 14].
                # Take the min/max as the inclusive page span.
                if isinstance(page_range, list) and len(page_range) > 0:
                    start_page = min(page_range)
                    end_page = max(page_range)

                    logger.debug(f"   Searching for images in pages {start_page}-{end_page} (from page_range: {page_range})")

                    # Get all images within this product's page range
                    images_result = self.supabase_client.client.table('document_images')\
                        .select('id, page_number')\
                        .eq('document_id', document_id)\
                        .gte('page_number', start_page)\
                        .lte('page_number', end_page)\
                        .execute()

                    images = images_result.data if images_result.data else []
                    logger.info(f"   Found {len(images)} images in pages {start_page}-{end_page}")

                    # Create relationships
                    for image in images:
                        try:
                            relationship = {
                                'product_id': product_id,
                                'image_id': image['id'],
                                'spatial_score': 1.0,  # High score for page-based matching
                                'caption_score': 0.0,
                                'clip_score': 0.0,
                                'overall_score': 1.0,
                                'confidence': 1.0,
                                'reasoning': 'page_proximity',  # replaces relationship_type
                                'metadata': {'page_based': True}
                            }

                            self.supabase_client.client.table('image_product_associations')\
                                .insert(relationship)\
                                .execute()

                            relationships_created += 1
                            logger.debug(f"   ✅ Linked image {image['id']} (page {image.get('page_number')}) to product {product_name}")

                        except Exception as e:
                            logger.warning(f"   ⚠️ Failed to create product-image relationship: {e}")

                products_processed += 1

            except Exception as e:
                logger.error(f"   ❌ Error processing product {product_id}: {e}", exc_info=True)
                continue

        logger.info(f"✅ Product-image relationship creation complete:")
        logger.info(f"   Products processed: {products_processed}/{len(product_ids)}")
        logger.info(f"   Products without page ranges: {products_without_pages}")
        logger.info(f"   Relationships created: {relationships_created}")

        return relationships_created

    async def create_all_relationships(
        self,
        document_id: str,
        product_ids: List[str],
        similarity_threshold: float = 0.5
    ) -> Dict[str, int]:
        """
        Create all relationships for a document.

        Args:
            document_id: Document ID
            product_ids: List of product IDs
            similarity_threshold: Minimum similarity for chunk-image relationships

        Returns:
            Dict with counts: {chunk_image_relationships, product_image_relationships}
        """
        logger.info(f"🔗 Creating product-image relationships for document {document_id}...")

        # chunk-image relationships are created by entity_linking_service using
        # page_proximity, not by this service. similarity_threshold is retained
        # for API compatibility but unused here.
        _ = similarity_threshold  # kept for caller compatibility

        product_image_count = await self.create_product_image_relationships(
            document_id=document_id,
            product_ids=product_ids
        )

        logger.info(f"✅ Product-image: {product_image_count}")

        return {
            'chunk_image_relationships': 0,  # Handled by entity_linking_service
            'product_image_relationships': product_image_count
        }


