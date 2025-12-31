"""
Chunk Relationship Service - Enhancement 5

Creates semantic relationships between document chunks to improve context retrieval.
This is a POST-PROCESSING service that runs AFTER chunks are stored.

Features:
- Finds semantically similar chunks using embeddings
- Creates relationships between related chunks
- Enables "show me everything about this product" queries
- Completely optional - doesn't affect existing pipeline

Safety:
- Runs AFTER all chunks are stored (no impact on main pipeline)
- Has fallback behavior (if fails, chunks still exist)
- Feature flag controlled (default: OFF)
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import numpy as np

from app.services.real_embeddings_service import RealEmbeddingsService
from app.services.supabase_client import get_supabase_client
from app.config import settings

logger = logging.getLogger(__name__)


class ChunkRelationshipService:
    """
    Creates and manages semantic relationships between document chunks.
    
    This service:
    1. Finds semantically similar chunks using embeddings
    2. Stores relationships in chunk_relationships table
    3. Enables better context retrieval for search
    4. Runs as optional post-processing step
    """
    
    # Similarity threshold for creating relationships
    SIMILARITY_THRESHOLD = 0.75  # High threshold for quality relationships
    
    # Maximum relationships per chunk
    MAX_RELATIONSHIPS_PER_CHUNK = 5
    
    def __init__(self, enabled: bool = False):
        """
        Initialize chunk relationship service.
        
        Args:
            enabled: Whether relationship mapping is enabled (default: False)
        """
        self.enabled = enabled
        self.embeddings_service = RealEmbeddingsService()
        self.logger = logger
    
    async def create_relationships(
        self,
        document_id: str,
        workspace_id: str,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create semantic relationships between chunks in a document.
        
        Args:
            document_id: Document ID to process
            workspace_id: Workspace ID
            job_id: Optional job ID for logging
            
        Returns:
            Dict with relationship creation stats
        """
        if not self.enabled:
            self.logger.info("⏭️ Chunk relationship mapping disabled (feature flag OFF)")
            return {
                "success": True,
                "enabled": False,
                "relationships_created": 0,
                "message": "Feature disabled"
            }
        
        try:
            self.logger.info(f"🔗 Creating chunk relationships for document {document_id}")
            
            # Get all chunks for this document
            supabase_client = get_supabase_client()
            chunks_result = supabase_client.client.table('document_chunks')\
                .select('id, content, metadata')\
                .eq('document_id', document_id)\
                .execute()
            
            if not chunks_result.data or len(chunks_result.data) < 2:
                self.logger.info(f"⏭️ Not enough chunks ({len(chunks_result.data) if chunks_result.data else 0}) to create relationships")
                return {
                    "success": True,
                    "relationships_created": 0,
                    "message": "Not enough chunks"
                }
            
            chunks = chunks_result.data
            self.logger.info(f"📊 Processing {len(chunks)} chunks for relationships")
            
            # Get embeddings for all chunks
            chunk_texts = [chunk['content'] for chunk in chunks]
            embeddings = await self._get_embeddings_batch(chunk_texts)
            
            # Find similar chunks and create relationships
            relationships = []
            for i, chunk in enumerate(chunks):
                similar_chunks = self._find_similar_chunks(
                    i, embeddings, chunks, self.SIMILARITY_THRESHOLD
                )
                
                # Create relationships for top N similar chunks
                for similar_idx, similarity in similar_chunks[:self.MAX_RELATIONSHIPS_PER_CHUNK]:
                    relationships.append({
                        'source_chunk_id': chunk['id'],
                        'target_chunk_id': chunks[similar_idx]['id'],
                        'relationship_type': 'semantic_similarity',
                        'similarity_score': float(similarity),
                        'document_id': document_id,
                        'workspace_id': workspace_id,
                        'created_at': datetime.utcnow().isoformat()
                    })
            
            # Store relationships in database
            if relationships:
                # Note: This requires chunk_relationships table to exist
                # If table doesn't exist, this will fail gracefully
                try:
                    result = supabase_client.client.table('chunk_relationships')\
                        .insert(relationships)\
                        .execute()
                    
                    self.logger.info(f"✅ Created {len(relationships)} chunk relationships")
                    return {
                        "success": True,
                        "relationships_created": len(relationships),
                        "chunks_processed": len(chunks)
                    }
                except Exception as db_error:
                    self.logger.warning(f"⚠️ Failed to store relationships (table may not exist): {db_error}")
                    return {
                        "success": False,
                        "error": "chunk_relationships table not found",
                        "relationships_created": 0
                    }
            else:
                self.logger.info("ℹ️ No relationships met similarity threshold")
                return {
                    "success": True,
                    "relationships_created": 0,
                    "message": "No similar chunks found"
                }
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create chunk relationships: {e}")
            # ✅ FALLBACK: Return success=False but don't raise exception
            # Chunks are already stored, relationships are optional
            return {
                "success": False,
                "error": str(e),
                "relationships_created": 0
            }

    async def _get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Get embeddings for a batch of texts.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        try:
            embeddings = []
            # Process in batches to avoid rate limits
            batch_size = 100
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = await self.embeddings_service.get_embeddings_batch(batch)
                embeddings.extend(batch_embeddings)

            return embeddings
        except Exception as e:
            self.logger.error(f"❌ Failed to get embeddings: {e}")
            raise

    def _find_similar_chunks(
        self,
        chunk_idx: int,
        embeddings: List[np.ndarray],
        chunks: List[Dict[str, Any]],
        threshold: float
    ) -> List[Tuple[int, float]]:
        """
        Find chunks similar to the given chunk.

        Args:
            chunk_idx: Index of the chunk to find similarities for
            embeddings: List of all chunk embeddings
            chunks: List of all chunks
            threshold: Similarity threshold

        Returns:
            List of (chunk_index, similarity_score) tuples
        """
        similar_chunks = []
        chunk_embedding = embeddings[chunk_idx]

        for i, other_embedding in enumerate(embeddings):
            if i == chunk_idx:
                continue  # Skip self

            # Calculate cosine similarity
            similarity = self._cosine_similarity(chunk_embedding, other_embedding)

            if similarity >= threshold:
                similar_chunks.append((i, similarity))

        # Sort by similarity (highest first)
        similar_chunks.sort(key=lambda x: x[1], reverse=True)

        return similar_chunks

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity score (0-1)
        """
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate cosine similarity: {e}")
            return 0.0


