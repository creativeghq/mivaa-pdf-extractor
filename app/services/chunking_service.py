"""
Chunking Service - Handles text chunking and text embeddings generation.

This service encapsulates all chunking-related operations:
1. Create semantic chunks from extracted text
2. Generate text embeddings for chunks
3. Create chunk-to-product relationships
"""

import logging
from typing import List, Dict, Any, Optional
from app.services.supabase_client import get_supabase_client
from app.services.real_embeddings_service import RealEmbeddingsService

logger = logging.getLogger(__name__)


class ChunkingService:
    """Service for handling text chunking and embeddings."""
    
    def __init__(self):
        self.supabase_client = get_supabase_client()
        self.embedding_service = RealEmbeddingsService()
    
    async def create_chunks_and_embeddings(
        self,
        document_id: str,
        workspace_id: str,
        extracted_text: str,
        product_ids: Optional[List[str]] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> Dict[str, Any]:
        """
        Create semantic chunks and generate text embeddings.
        
        Args:
            document_id: Document ID
            workspace_id: Workspace ID
            extracted_text: Full extracted text from PDF
            product_ids: Optional list of product IDs for relationships
            chunk_size: Size of each chunk in tokens
            chunk_overlap: Overlap between chunks
            
        Returns:
            Dict with counts: {chunks_created, embeddings_generated, relationships_created}
        """
        logger.info(f"📝 Creating chunks for document {document_id}...")
        
        # Import chunking utilities
        from app.services.chunking_utils import create_semantic_chunks
        
        # Create semantic chunks
        chunks = create_semantic_chunks(
            text=extracted_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        logger.info(f"   Created {len(chunks)} semantic chunks")
        
        chunks_created = 0
        embeddings_generated = 0
        relationships_created = 0
        
        # Process each chunk
        for idx, chunk_data in enumerate(chunks):
            try:
                # Save chunk to database
                chunk_record = {
                    'document_id': document_id,
                    'workspace_id': workspace_id,
                    'content': chunk_data['text'],
                    'chunk_index': idx,
                    'metadata': {
                        'start_char': chunk_data.get('start_char', 0),
                        'end_char': chunk_data.get('end_char', 0),
                        'page_number': chunk_data.get('page_number', 1)
                    }
                }
                
                result = self.supabase_client.client.table('chunks').insert(chunk_record).execute()
                
                if result.data and len(result.data) > 0:
                    chunk_id = result.data[0]['id']
                    chunks_created += 1
                    
                    # Generate text embedding
                    embedding_result = await self.embedding_service.generate_all_embeddings(
                        entity_id=chunk_id,
                        entity_type="chunk",
                        text_content=chunk_data['text'],
                        image_data=None,
                        material_properties={}
                    )
                    
                    if embedding_result and embedding_result.get('success'):
                        embeddings = embedding_result.get('embeddings', {})
                        text_embedding = embeddings.get('text_512')
                        
                        if text_embedding:
                            # Save text embedding to chunks table
                            self.supabase_client.client.table('chunks')\
                                .update({'embedding': text_embedding})\
                                .eq('id', chunk_id)\
                                .execute()
                            
                            embeddings_generated += 1
                            logger.debug(f"   ✅ Generated text embedding for chunk {idx + 1}/{len(chunks)}")
                    
                    # Create chunk-to-product relationships
                    if product_ids:
                        for product_id in product_ids:
                            try:
                                relationship = {
                                    'chunk_id': chunk_id,
                                    'product_id': product_id,
                                    'relevance_score': 1.0  # Default score, can be improved with AI
                                }
                                
                                self.supabase_client.client.table('chunk_product_relationships')\
                                    .insert(relationship)\
                                    .execute()
                                
                                relationships_created += 1
                            
                            except Exception as rel_error:
                                logger.warning(f"   ⚠️ Failed to create chunk-product relationship: {rel_error}")
                
            except Exception as e:
                logger.error(f"   ❌ Error processing chunk {idx + 1}: {e}")
                continue
        
        logger.info(f"✅ Chunking complete:")
        logger.info(f"   Chunks created: {chunks_created}")
        logger.info(f"   Text embeddings generated: {embeddings_generated}")
        logger.info(f"   Chunk-product relationships: {relationships_created}")
        
        return {
            'chunks_created': chunks_created,
            'embeddings_generated': embeddings_generated,
            'relationships_created': relationships_created
        }

