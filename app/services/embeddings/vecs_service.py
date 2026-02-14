"""
VECS Service for managing vector embeddings with Supabase.

This service uses the vecs library (Supabase's recommended approach) for:
- Storing SLIG image embeddings (768D)
- Storing understanding embeddings (1024D) from Voyage AI
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
        """Build Supabase connection string for vecs.

        Supabase Pooler Connection (IPv4):
        Format: postgresql://postgres.{project_id}:{password}@aws-0-{region}.pooler.supabase.com:5432/postgres

        CRITICAL: Use port 5432 (not 6543) for pooler with database password.
        """
        db_password = os.getenv('SUPABASE_DB_PASSWORD')
        if not db_password:
            raise ValueError(
                "SUPABASE_DB_PASSWORD not found in environment. "
                "VECS requires the actual database password, not the service role key. "
                "Please set SUPABASE_DB_PASSWORD in systemd environment variables."
            )

        # Supabase pooler connection for IPv4 (port 5432, not 6543)
        project_id = os.getenv('SUPABASE_PROJECT_ID', 'bgbavxtjlbvgplozizxu')
        db_host = "aws-0-eu-west-3.pooler.supabase.com"
        db_user = f"postgres.{project_id}"

        connection_string = f"postgresql://{db_user}:{db_password}@{db_host}:5432/postgres"
        logger.info(f"Using database password for VECS connection (pooler IPv4 mode, port 5432)")

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
            name: Collection name (e.g., 'image_slig_embeddings')
            dimension: Vector dimension (e.g., 768 for SLIG)
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
        siglip_embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Upsert a single image SigLIP embedding.

        Args:
            image_id: Image UUID
            siglip_embedding: 768D SLIG embedding
            metadata: Optional metadata (document_id, page_number, quality_score, etc.)

        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.get_or_create_collection(
                name="image_slig_embeddings",
                dimension=768
            )
            
            # Prepare metadata
            meta = metadata or {}
            
            # Upsert single record
            collection.upsert(
                records=[(image_id, siglip_embedding, meta)]
            )

            logger.debug(f"✅ Upserted SigLIP embedding for image {image_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upsert embedding for image {image_id}: {e}")
            return False
    
    async def batch_upsert_image_embeddings(
        self,
        records: List[Tuple[str, List[float], Dict[str, Any]]]
    ) -> int:
        """
        Batch upsert multiple image SigLIP embeddings.

        Args:
            records: List of (image_id, siglip_embedding, metadata) tuples

        Returns:
            Number of successfully upserted records
        """
        try:
            collection = self.get_or_create_collection(
                name="image_slig_embeddings",
                dimension=768
            )

            # Batch upsert
            collection.upsert(records=records)

            logger.info(f"✅ Batch upserted {len(records)} SigLIP embeddings")
            return len(records)
            
        except Exception as e:
            logger.error(f"❌ Batch upsert failed: {e}")
            return 0

    async def upsert_specialized_embeddings(
        self,
        image_id: str,
        embeddings: Dict[str, List[float]],
        metadata: Dict[str, Any]
    ) -> Dict[str, bool]:
        """
        Upsert specialized text-guided SigLIP embeddings to their respective collections.

        Args:
            image_id: Image UUID
            embeddings: Dict with keys: color, texture, style, material (each 768D)
            metadata: Metadata to store with each embedding

        Returns:
            Dict mapping collection name to success status
        """
        results = {}

        collection_mapping = {
            "color": "image_color_embeddings",
            "texture": "image_texture_embeddings",
            "style": "image_style_embeddings",
            "material": "image_material_embeddings"
        }

        for embedding_type, collection_name in collection_mapping.items():
            if embedding_type in embeddings and embeddings[embedding_type]:
                try:
                    collection = self.get_or_create_collection(
                        name=collection_name,
                        dimension=768  # Updated to 768D for SLIG
                    )

                    # Upsert single record
                    collection.upsert(records=[(image_id, embeddings[embedding_type], metadata)])

                    results[collection_name] = True
                    logger.debug(f"✅ Upserted {embedding_type} embedding (768D) to '{collection_name}'")

                except Exception as e:
                    logger.error(f"❌ Failed to upsert {embedding_type} embedding: {e}")
                    results[collection_name] = False
            else:
                results[collection_name] = False

        success_count = sum(1 for v in results.values() if v)
        logger.info(f"✅ Upserted {success_count}/4 specialized embeddings for image {image_id}")

        return results

    async def upsert_understanding_embedding(
        self,
        image_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Upsert a vision-understanding embedding (Qwen analysis → Voyage AI 1024D).

        Args:
            image_id: Image UUID
            embedding: 1024D understanding embedding from Voyage AI
            metadata: Optional metadata (document_id, workspace_id, page_number, etc.)

        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.get_or_create_collection(
                name="image_understanding_embeddings",
                dimension=1024
            )

            meta = metadata or {}

            collection.upsert(
                records=[(image_id, embedding, meta)]
            )

            logger.debug(f"✅ Upserted understanding embedding for image {image_id}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to upsert understanding embedding for image {image_id}: {e}")
            return False

    async def search_understanding_embeddings(
        self,
        query_embedding: List[float],
        limit: int = 10,
        workspace_id: Optional[str] = None,
        document_id: Optional[str] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar images using vision-understanding embeddings (1024D).

        Args:
            query_embedding: 1024D query embedding from Voyage AI
            limit: Maximum number of results
            workspace_id: Optional workspace ID filter
            document_id: Optional document ID filter
            include_metadata: Whether to include metadata in results

        Returns:
            List of similar images with similarity scores
        """
        try:
            collection = self.get_or_create_collection(
                name="image_understanding_embeddings",
                dimension=1024
            )

            # Build filters
            filters = None
            if workspace_id:
                filters = {"workspace_id": {"$eq": workspace_id}}
                if document_id:
                    filters["document_id"] = {"$eq": document_id}
            elif document_id:
                filters = {"document_id": {"$eq": document_id}}

            # Run in thread pool for true async parallelism
            import asyncio
            results = await asyncio.to_thread(
                collection.query,
                data=query_embedding,
                limit=limit,
                filters=filters,
                include_value=True,
                include_metadata=include_metadata
            )

            formatted_results = []
            for result_tuple in results:
                if include_metadata:
                    image_id, distance, metadata = result_tuple
                else:
                    image_id, distance = result_tuple
                    metadata = None

                result = {
                    "image_id": image_id,
                    "similarity_score": 1 - distance,
                    "distance": distance,
                    "search_type": "understanding_similarity"
                }
                if include_metadata and metadata:
                    result["metadata"] = metadata
                formatted_results.append(result)

            logger.info(f"✅ Found {len(formatted_results)} similar images using understanding embeddings")
            return formatted_results

        except Exception as e:
            logger.error(f"❌ Understanding embedding search failed: {e}")
            return []

    async def search_similar_images(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar images using SigLIP embeddings.

        Args:
            query_embedding: 768D query embedding
            limit: Maximum number of results
            filters: Optional metadata filters (e.g., {"document_id": {"$eq": "uuid"}})
            include_metadata: Whether to include metadata in results

        Returns:
            List of search results with image_id, distance, and metadata
        """
        try:
            collection = self.get_or_create_collection(
                name="image_slig_embeddings",
                dimension=768
            )

            # Query collection - Run in thread pool for true async parallelism
            import asyncio
            results = await asyncio.to_thread(
                collection.query,
                data=query_embedding,
                limit=limit,
                filters=filters,
                include_value=True,  # Return distance for similarity scoring
                include_metadata=include_metadata
            )

            # Format results - handle different return formats based on include_value/include_metadata
            formatted_results = []
            for result_tuple in results:
                # With include_value=True and include_metadata=True: (id, distance, metadata)
                # With include_value=True and include_metadata=False: (id, distance)
                # With include_value=False and include_metadata=True: (id, metadata)
                # With include_value=False and include_metadata=False: just id (string)
                if include_metadata:
                    image_id, distance, metadata = result_tuple
                else:
                    image_id, distance = result_tuple
                    metadata = None

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

    async def count_embeddings(
        self,
        workspace_id: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> int:
        """
        Count embeddings in the VECS collection with optional filtering.

        Args:
            workspace_id: Optional workspace ID to filter by
            document_id: Optional document ID to filter by

        Returns:
            Number of embeddings matching the filters
        """
        try:
            collection = self.get_or_create_collection(
                name="image_slig_embeddings",
                dimension=768
            )

            # Build filters
            filters = None
            if workspace_id:
                filters = {"workspace_id": {"$eq": workspace_id}}
                if document_id:
                    filters["document_id"] = {"$eq": document_id}
            elif document_id:
                filters = {"document_id": {"$eq": document_id}}

            # Query with large limit to count all
            # Note: vecs doesn't have a direct count method, so we query and count results
            results = collection.query(
                data=[0.0] * 768,  # Dummy query vector
                limit=100000,  # Large limit to get all embeddings
                filters=filters,
                include_value=False,
                include_metadata=False
            )

            count = len(results)
            logger.info(f"✅ Counted {count} embeddings (workspace_id={workspace_id}, document_id={document_id})")
            return count

        except Exception as e:
            logger.error(f"❌ Failed to count embeddings: {e}")
            return 0

    async def search_specialized_embeddings(
        self,
        query_embedding: List[float],
        embedding_type: str,  # 'color', 'texture', 'style', 'material'
        limit: int = 10,
        workspace_id: Optional[str] = None,
        document_id: Optional[str] = None,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Search for similar images using text-guided specialized SigLIP embeddings.

        Args:
            query_embedding: Query embedding vector (768D)
            embedding_type: Type of embedding ('color', 'texture', 'style', 'material')
            limit: Maximum number of results
            workspace_id: Optional workspace ID filter
            document_id: Optional document ID filter
            include_metadata: Whether to include metadata in results

        Returns:
            List of similar images with similarity scores
        """
        try:
            collection_mapping = {
                "color": "image_color_embeddings",
                "texture": "image_texture_embeddings",
                "style": "image_style_embeddings",
                "material": "image_material_embeddings"
            }

            if embedding_type not in collection_mapping:
                logger.error(f"Invalid embedding type: {embedding_type}")
                return []

            collection_name = collection_mapping[embedding_type]
            collection = self.get_or_create_collection(
                name=collection_name,
                dimension=768  # Updated to 768D for SLIG
            )

            # Build filters
            filters = None
            if workspace_id:
                filters = {"workspace_id": {"$eq": workspace_id}}
                if document_id:
                    filters["document_id"] = {"$eq": document_id}
            elif document_id:
                filters = {"document_id": {"$eq": document_id}}

            # Search - Run in thread pool for true async parallelism
            # VECS collection.query() is synchronous, so we need to run it in a thread
            import asyncio
            results = await asyncio.to_thread(
                collection.query,
                data=query_embedding,
                limit=limit,
                filters=filters,
                include_value=False,
                include_metadata=include_metadata
            )

            # Format results
            formatted_results = []
            for image_id, distance, metadata in results:
                # Convert distance to similarity score (0-1)
                similarity_score = 1.0 / (1.0 + distance)

                result = {
                    "image_id": image_id,
                    "similarity_score": similarity_score,
                    "distance": distance,
                    "search_type": f"{embedding_type}_similarity"
                }

                if include_metadata and metadata:
                    result["metadata"] = metadata

                formatted_results.append(result)

            logger.info(f"✅ Found {len(formatted_results)} similar images using {embedding_type} embeddings")
            return formatted_results

        except Exception as e:
            logger.error(f"❌ Specialized search ({embedding_type}) failed: {e}")
            return []

    async def search_all_collections(
        self,
        visual_query_embedding: List[float],
        specialized_query_embeddings: Optional[Dict[str, List[float]]] = None,
        understanding_query_embedding: Optional[List[float]] = None,
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Search all VECS collections in parallel and return combined results with multi-vector scores.

        This enables true multi-vector search by querying:
        - image_slig_embeddings (primary visual, 768D)
        - image_understanding_embeddings (vision-understanding, 1024D)
        - image_color_embeddings (768D)
        - image_texture_embeddings (768D)
        - image_style_embeddings (768D)
        - image_material_embeddings (768D)

        Args:
            visual_query_embedding: Primary 768D visual query embedding
            specialized_query_embeddings: Optional dict with keys: color, texture, style, material
                                         If not provided, visual_query_embedding is used for all
            understanding_query_embedding: Optional 1024D understanding query embedding from Voyage AI
            limit: Maximum results per collection
            filters: Optional metadata filters
            include_metadata: Whether to include metadata

        Returns:
            Dict with:
            - results: List of combined results with all similarity scores
            - collection_stats: Stats per collection
        """
        import asyncio

        # Use visual embedding for specialized if not provided
        if specialized_query_embeddings is None:
            specialized_query_embeddings = {}

        # Define search tasks
        async def search_visual():
            return await self.search_similar_images(
                query_embedding=visual_query_embedding,
                limit=limit,
                filters=filters,
                include_metadata=include_metadata
            )

        async def search_understanding():
            if understanding_query_embedding is None:
                return []
            return await self.search_understanding_embeddings(
                query_embedding=understanding_query_embedding,
                limit=limit,
                workspace_id=filters.get('workspace_id', {}).get('$eq') if filters else None,
                document_id=filters.get('document_id', {}).get('$eq') if filters else None,
                include_metadata=include_metadata
            )

        async def search_specialized(embedding_type: str, query_emb: List[float]):
            return await self.search_specialized_embeddings(
                query_embedding=query_emb,
                embedding_type=embedding_type,
                limit=limit,
                workspace_id=filters.get('workspace_id', {}).get('$eq') if filters else None,
                document_id=filters.get('document_id', {}).get('$eq') if filters else None,
                include_metadata=include_metadata
            )

        try:
            # Run all searches in parallel
            tasks = [search_visual(), search_understanding()]

            # Add specialized searches if embeddings provided
            for emb_type in ['color', 'texture', 'style', 'material']:
                query_emb = specialized_query_embeddings.get(emb_type, visual_query_embedding)
                tasks.append(search_specialized(emb_type, query_emb))

            # Execute all searches in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Parse results
            visual_results = results[0] if not isinstance(results[0], Exception) else []
            understanding_results = results[1] if not isinstance(results[1], Exception) else []
            color_results = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else []
            texture_results = results[3] if len(results) > 3 and not isinstance(results[3], Exception) else []
            style_results = results[4] if len(results) > 4 and not isinstance(results[4], Exception) else []
            material_results = results[5] if len(results) > 5 and not isinstance(results[5], Exception) else []

            # Build lookup maps for all scores
            understanding_scores = {r['image_id']: r['similarity_score'] for r in understanding_results}
            color_scores = {r['image_id']: r['similarity_score'] for r in color_results}
            texture_scores = {r['image_id']: r['similarity_score'] for r in texture_results}
            style_scores = {r['image_id']: r['similarity_score'] for r in style_results}
            material_scores = {r['image_id']: r['similarity_score'] for r in material_results}

            # Collect all unique image IDs from all collections
            all_image_ids = set()
            for result in visual_results:
                all_image_ids.add(result['image_id'])
            for score_map in [understanding_scores, color_scores, texture_scores, style_scores, material_scores]:
                all_image_ids.update(score_map.keys())

            # Build visual score lookup
            visual_scores_map = {r['image_id']: r for r in visual_results}

            # Determine weights based on whether understanding embeddings are available
            has_understanding = bool(understanding_query_embedding and understanding_results)

            if has_understanding:
                # 6-vector weights: visual=25%, understanding=20%, color=14%, texture=14%, style=14%, material=13%
                w_visual = 0.25
                w_understanding = 0.20
                w_color = 0.14
                w_texture = 0.14
                w_style = 0.14
                w_material = 0.13
            else:
                # Fallback to 5-vector weights when no understanding embedding
                w_visual = 0.35
                w_understanding = 0.0
                w_color = 0.1625
                w_texture = 0.1625
                w_style = 0.1625
                w_material = 0.1625

            # Combine results from all collections
            combined_results = []
            for image_id in all_image_ids:
                visual_data = visual_scores_map.get(image_id)
                visual_score = visual_data.get('similarity_score', 0.0) if visual_data else 0.0
                fallback = visual_score * 0.8

                understanding_score = understanding_scores.get(image_id, 0.0)
                color_score = color_scores.get(image_id, fallback)
                texture_score = texture_scores.get(image_id, fallback)
                style_score = style_scores.get(image_id, fallback)
                material_score = material_scores.get(image_id, fallback)

                combined_score = (
                    w_visual * visual_score +
                    w_understanding * understanding_score +
                    w_color * color_score +
                    w_texture * texture_score +
                    w_style * style_score +
                    w_material * material_score
                )

                combined_result = {
                    'image_id': image_id,
                    'similarity_score': visual_score,
                    'combined_score': combined_score,
                    'scores': {
                        'visual': visual_score,
                        'understanding': understanding_score,
                        'color': color_score,
                        'texture': texture_score,
                        'style': style_score,
                        'material': material_score
                    },
                    'metadata': visual_data.get('metadata', {}) if visual_data else {}
                }
                combined_results.append(combined_result)

            # Sort by combined score
            combined_results.sort(key=lambda x: x['combined_score'], reverse=True)

            # Trim to limit
            combined_results = combined_results[:limit]

            # Stats
            collection_stats = {
                'visual_count': len(visual_results),
                'understanding_count': len(understanding_results),
                'color_count': len(color_results),
                'texture_count': len(texture_results),
                'style_count': len(style_results),
                'material_count': len(material_results)
            }

            logger.info(
                f"✅ Multi-vector search complete: {len(combined_results)} results "
                f"(visual={len(visual_results)}, understanding={len(understanding_results)}, "
                f"color={len(color_results)}, texture={len(texture_results)}, "
                f"style={len(style_results)}, material={len(material_results)})"
            )

            return {
                'results': combined_results,
                'collection_stats': collection_stats
            }

        except Exception as e:
            logger.error(f"❌ Multi-vector search failed: {e}")
            return {
                'results': [],
                'collection_stats': {},
                'error': str(e)
            }

    async def delete_image_embedding(self, image_id: str) -> bool:
        """
        Delete an image embedding from all collections (visual, understanding, specialized).

        Args:
            image_id: Image UUID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete from primary visual collection
            collection = self.get_or_create_collection(
                name="image_slig_embeddings",
                dimension=768
            )
            collection.delete(ids=[image_id])

            # Delete from understanding collection
            try:
                understanding_col = self.get_or_create_collection(
                    name="image_understanding_embeddings",
                    dimension=1024
                )
                understanding_col.delete(ids=[image_id])
            except Exception:
                pass  # Collection may not exist yet

            # Delete from specialized collections
            for col_name, dim in [
                ("image_color_embeddings", 768),
                ("image_texture_embeddings", 768),
                ("image_style_embeddings", 768),
                ("image_material_embeddings", 768),
            ]:
                try:
                    col = self.get_or_create_collection(name=col_name, dimension=dim)
                    col.delete(ids=[image_id])
                except Exception:
                    pass

            logger.debug(f"✅ Deleted embeddings for image {image_id} from all collections")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to delete embedding for image {image_id}: {e}")
            return False

    async def delete_document_embeddings(self, document_id: str) -> int:
        """
        Delete all image embeddings for a document from all collections.

        Args:
            document_id: Document UUID

        Returns:
            Number of embeddings deleted from the primary collection
        """
        try:
            collection = self.get_or_create_collection(
                name="image_slig_embeddings",
                dimension=768
            )

            # Query to get all image IDs for this document
            # Note: vecs doesn't support delete by filter, so we need to query first
            results = collection.query(
                data=[0.0] * 768,  # Dummy query
                limit=10000,  # Large limit to get all
                filters={"document_id": {"$eq": document_id}},
                include_value=False,
                include_metadata=False
            )

            image_ids = [image_id for image_id, _, _ in results]

            if image_ids:
                collection.delete(ids=image_ids)

                # Also delete from understanding collection
                try:
                    understanding_col = self.get_or_create_collection(
                        name="image_understanding_embeddings",
                        dimension=1024
                    )
                    understanding_col.delete(ids=image_ids)
                except Exception:
                    pass

                # Delete from specialized collections
                for col_name, dim in [
                    ("image_color_embeddings", 768),
                    ("image_texture_embeddings", 768),
                    ("image_style_embeddings", 768),
                    ("image_material_embeddings", 768),
                ]:
                    try:
                        col = self.get_or_create_collection(name=col_name, dimension=dim)
                        col.delete(ids=image_ids)
                    except Exception:
                        pass

                logger.info(f"✅ Deleted {len(image_ids)} embeddings for document {document_id} from all collections")
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


