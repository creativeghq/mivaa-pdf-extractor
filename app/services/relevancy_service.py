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
import numpy as np
from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class RelevancyService:
    """Service for managing entity relationships and relevancy scores."""
    
    def __init__(self):
        self.supabase_client = get_supabase_client()
    
    def calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
        
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    async def create_chunk_image_relationships(
        self,
        document_id: str,
        similarity_threshold: float = 0.5
    ) -> int:
        """
        Create chunk-to-image relationships based on embedding similarity.
        
        Args:
            document_id: Document ID
            similarity_threshold: Minimum similarity score to create relationship
            
        Returns:
            Number of relationships created
        """
        logger.info(f"🔗 Creating chunk-image relationships for document {document_id}...")
        
        # Get all chunks with embeddings
        chunks_result = self.supabase_client.client.table('chunks')\
            .select('id, embedding')\
            .eq('document_id', document_id)\
            .not_.is_('embedding', 'null')\
            .execute()
        
        chunks = chunks_result.data if chunks_result.data else []
        
        # Get all images with CLIP embeddings
        images_result = self.supabase_client.client.table('document_images')\
            .select('id, visual_clip_embedding_512')\
            .eq('document_id', document_id)\
            .not_.is_('visual_clip_embedding_512', 'null')\
            .execute()
        
        images = images_result.data if images_result.data else []
        
        logger.info(f"   Found {len(chunks)} chunks and {len(images)} images with embeddings")
        
        relationships_created = 0
        
        # Create relationships based on similarity
        for chunk in chunks:
            chunk_embedding = chunk['embedding']
            
            for image in images:
                image_embedding = image['visual_clip_embedding_512']
                
                # Calculate similarity
                similarity = self.calculate_cosine_similarity(chunk_embedding, image_embedding)
                
                if similarity >= similarity_threshold:
                    try:
                        relationship = {
                            'chunk_id': chunk['id'],
                            'image_id': image['id'],
                            'relevance_score': similarity
                        }
                        
                        self.supabase_client.client.table('chunk_image_relationships')\
                            .insert(relationship)\
                            .execute()
                        
                        relationships_created += 1
                        logger.debug(f"   ✅ Created chunk-image relationship (similarity: {similarity:.3f})")
                    
                    except Exception as e:
                        logger.warning(f"   ⚠️ Failed to create chunk-image relationship: {e}")
        
        logger.info(f"✅ Created {relationships_created} chunk-image relationships")
        
        return relationships_created
    
    async def create_product_image_relationships(
        self,
        document_id: str,
        product_ids: List[str]
    ) -> int:
        """
        Create product-to-image relationships based on page numbers and metadata.
        
        Args:
            document_id: Document ID
            product_ids: List of product IDs
            
        Returns:
            Number of relationships created
        """
        logger.info(f"🔗 Creating product-image relationships for {len(product_ids)} products...")
        
        relationships_created = 0
        
        for product_id in product_ids:
            try:
                # Get product metadata (page ranges)
                product_result = self.supabase_client.client.table('products')\
                    .select('id, metadata')\
                    .eq('id', product_id)\
                    .execute()
                
                if not product_result.data or len(product_result.data) == 0:
                    continue
                
                product = product_result.data[0]
                metadata = product.get('metadata', {})
                page_ranges = metadata.get('page_ranges', [])
                
                if not page_ranges:
                    continue
                
                # Get images within product page ranges
                for page_range in page_ranges:
                    start_page = page_range.get('start', 0)
                    end_page = page_range.get('end', 0)
                    
                    images_result = self.supabase_client.client.table('document_images')\
                        .select('id')\
                        .eq('document_id', document_id)\
                        .gte('metadata->>page_number', start_page)\
                        .lte('metadata->>page_number', end_page)\
                        .execute()

                    images = images_result.data if images_result.data else []

                    # Create relationships
                    for image in images:
                        try:
                            relationship = {
                                'product_id': product_id,
                                'image_id': image['id'],
                                'relevance_score': 1.0  # High score for page-based matching
                            }

                            self.supabase_client.client.table('product_image_relationships')\
                                .insert(relationship)\
                                .execute()

                            relationships_created += 1

                        except Exception as e:
                            logger.warning(f"   ⚠️ Failed to create product-image relationship: {e}")

            except Exception as e:
                logger.error(f"   ❌ Error processing product {product_id}: {e}")
                continue

        logger.info(f"✅ Created {relationships_created} product-image relationships")

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
        logger.info(f"🔗 Creating all relationships for document {document_id}...")

        # Create chunk-image relationships
        chunk_image_count = await self.create_chunk_image_relationships(
            document_id=document_id,
            similarity_threshold=similarity_threshold
        )

        # Create product-image relationships
        product_image_count = await self.create_product_image_relationships(
            document_id=document_id,
            product_ids=product_ids
        )

        logger.info(f"✅ All relationships created:")
        logger.info(f"   Chunk-image: {chunk_image_count}")
        logger.info(f"   Product-image: {product_image_count}")

        return {
            'chunk_image_relationships': chunk_image_count,
            'product_image_relationships': product_image_count
        }

