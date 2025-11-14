"""
VECS Service for managing vector embeddings with Supabase.

This service uses the vecs library (Supabase's recommended approach) for:
- Storing CLIP image embeddings (512D)
- Fast similarity search with automatic indexing
- Metadata filtering
- Batch operations

Replaces manual SQL queries with <=> operator.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import vecs
from vecs.collection import Collection, IndexMethod

logger = logging.getLogger(__name__)


class VecsService:
    """Service for managing vector embeddings using Supabase vecs library."""

    _instance = None

    def __init__(self):
        """Initialize VECS client with Supabase connection."""
        self._client = None
        self._collections: Dict[str, Collection] = {}
        
    def _get_connection_string(self) -> str:
        """Build Supabase connection string for vecs."""
        # Try to get database password first (preferred)
        db_password = os.getenv('SUPABASE_DB_PASSWORD')

        if db_password:
            # Use database password with connection pooler
            project_id = os.getenv('SUPABASE_PROJECT_ID', 'bgbavxtjlbvgplozizxu')
            connection_string = f"postgresql://postgres.{project_id}:{db_password}@aws-0-eu-west-3.pooler.supabase.com:6543/postgres"
            logger.info("Using database password for VECS connection")
        else:
            # Fallback: Use direct connection with service role key
            # This requires the database to accept JWT authentication
            service_role_key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
            if not service_role_key:
                raise ValueError("Neither SUPABASE_DB_PASSWORD nor SUPABASE_SERVICE_ROLE_KEY found in environment")

            project_id = os.getenv('SUPABASE_PROJECT_ID', 'bgbavxtjlbvgplozizxu')
            # Use direct connection (port 5432) with service role key
            connection_string = f"postgresql://postgres:{service_role_key}@db.{project_id}.supabase.co:5432/postgres"
            logger.info("Using service role key for VECS connection (direct mode)")

        return connection_string
    
    def _get_client(self) -> vecs.Client:
        """Get or create vecs client."""
        if self._client is None:
            connection_string = self._get_connection_string()
            self._client = vecs.create_client(connection_string)
            logger.info("✅ VECS client initialized")
        return self._client
    
    def get_or_create_collection(
        self,
        name: str,
        dimension: int,
        create_index: bool = True,
        index_method: str = "hnsw"
    ) -> Collection:
        """
        Get or create a vector collection with optimized indexing.

        Args:
            name: Collection name (e.g., 'image_clip_embeddings')
            dimension: Vector dimension (e.g., 512 for CLIP)
            create_index: Whether to create index for fast search
            index_method: Index method - 'hnsw' (fast, approximate) or 'ivfflat' (balanced)

        Returns:
            VECS Collection instance
        """
        if name in self._collections:
            return self._collections[name]

        try:
            client = self._get_client()

            # Get or create collection
            collection = client.get_or_create_collection(
                name=name,
                dimension=dimension
            )

            # ✅ FIX: Create HNSW index for fast similarity search (if not exists)
            if create_index:
                try:
                    # HNSW parameters for optimal performance:
                    # - m=16: Number of connections per layer (higher = better recall, more memory)
                    # - ef_construction=64: Size of dynamic candidate list (higher = better index quality)
                    if index_method == "hnsw":
                        collection.create_index(method=IndexMethod.hnsw, m=16, ef_construction=64)
                        logger.info(f"✅ Created HNSW index for collection '{name}' (m=16, ef_construction=64)")
                    else:
                        collection.create_index(method=IndexMethod.ivfflat)
                        logger.info(f"✅ Created IVFFlat index for collection '{name}'")
                except Exception as index_error:
                    # Index might already exist
                    logger.debug(f"Index creation skipped for '{name}': {index_error}")

            self._collections[name] = collection
            logger.info(f"✅ Collection '{name}' ready (dimension={dimension}, index={index_method})")

            return collection

        except Exception as e:
            logger.error(f"❌ Failed to get/create collection '{name}': {e}")
            raise
    
    async def upsert_image_embedding(
        self,
        image_id: str,
        clip_embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Upsert a single image CLIP embedding.
        
        Args:
            image_id: Image UUID
            clip_embedding: 512D CLIP embedding
            metadata: Optional metadata (document_id, page_number, quality_score, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.get_or_create_collection(
                name="image_clip_embeddings",
                dimension=512
            )
            
            # Prepare metadata
            meta = metadata or {}
            
            # Upsert single record
            collection.upsert(
                records=[(image_id, clip_embedding, meta)]
            )
            
            logger.debug(f"✅ Upserted CLIP embedding for image {image_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upsert embedding for image {image_id}: {e}")
            return False
    
    async def batch_upsert_image_embeddings(
        self,
        records: List[Tuple[str, List[float], Dict[str, Any]]]
    ) -> int:
        """
        Batch upsert multiple image CLIP embeddings.
        
        Args:
            records: List of (image_id, clip_embedding, metadata) tuples
            
        Returns:
            Number of successfully upserted records
        """
        try:
            collection = self.get_or_create_collection(
                name="image_clip_embeddings",
                dimension=512
            )
            
            # Batch upsert
            collection.upsert(records=records)
            
            logger.info(f"✅ Batch upserted {len(records)} CLIP embeddings")
            return len(records)
            
        except Exception as e:
            logger.error(f"❌ Batch upsert failed: {e}")
            return 0

    async def search_similar_images(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar images using CLIP embeddings.

        Args:
            query_embedding: 512D query embedding
            limit: Maximum number of results
            filters: Optional metadata filters (e.g., {"document_id": {"$eq": "uuid"}})
            include_metadata: Whether to include metadata in results

        Returns:
            List of search results with image_id, distance, and metadata
        """
        try:
            collection = self.get_or_create_collection(
                name="image_clip_embeddings",
                dimension=512
            )

            # Query collection
            results = collection.query(
                data=query_embedding,
                limit=limit,
                filters=filters,
                include_value=False,  # Don't return the embedding vector
                include_metadata=include_metadata
            )

            # Format results
            formatted_results = []
            for image_id, distance, metadata in results:
                result = {
                    "image_id": image_id,
                    "similarity_score": 1 - distance,  # Convert distance to similarity
                    "distance": distance
                }
                if include_metadata and metadata:
                    result["metadata"] = metadata
                formatted_results.append(result)

            logger.info(f"✅ Found {len(formatted_results)} similar images")
            return formatted_results

        except Exception as e:
            logger.error(f"❌ Image similarity search failed: {e}")
            return []

    async def delete_image_embedding(self, image_id: str) -> bool:
        """
        Delete an image embedding from the collection.

        Args:
            image_id: Image UUID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.get_or_create_collection(
                name="image_clip_embeddings",
                dimension=512
            )

            collection.delete(ids=[image_id])

            logger.debug(f"✅ Deleted embedding for image {image_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete embedding for image {image_id}: {e}")
            return False

    async def delete_document_embeddings(self, document_id: str) -> int:
        """
        Delete all image embeddings for a document.

        Args:
            document_id: Document UUID

        Returns:
            Number of embeddings deleted
        """
        try:
            collection = self.get_or_create_collection(
                name="image_clip_embeddings",
                dimension=512
            )

            # Query to get all image IDs for this document
            # Note: vecs doesn't support delete by filter, so we need to query first
            results = collection.query(
                data=[0.0] * 512,  # Dummy query
                limit=10000,  # Large limit to get all
                filters={"document_id": {"$eq": document_id}},
                include_value=False,
                include_metadata=False
            )

            image_ids = [image_id for image_id, _, _ in results]

            if image_ids:
                collection.delete(ids=image_ids)
                logger.info(f"✅ Deleted {len(image_ids)} embeddings for document {document_id}")
                return len(image_ids)
            else:
                logger.info(f"No embeddings found for document {document_id}")
                return 0

        except Exception as e:
            logger.error(f"❌ Failed to delete embeddings for document {document_id}: {e}")
            return 0


# Global instance
_vecs_service: Optional[VecsService] = None


def get_vecs_service() -> VecsService:
    """Get or create global VecsService instance."""
    global _vecs_service
    if _vecs_service is None:
        _vecs_service = VecsService()
    return _vecs_service

