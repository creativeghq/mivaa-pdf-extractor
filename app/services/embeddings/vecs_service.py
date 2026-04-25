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
        self._supabase_rest = None  # lazy-init Supabase REST client for boolean flag updates

    def _get_supabase_rest(self):
        """Lazy init of the Supabase REST client used for updating
        document_images.has_*_slig boolean flags after each VECS upsert."""
        if self._supabase_rest is None:
            from app.services.core.supabase_client import get_supabase_client
            self._supabase_rest = get_supabase_client()
        return self._supabase_rest

    def _set_image_flag(self, image_id: str, flag_column: str) -> None:
        """Set a presence flag (e.g. has_slig_embedding=true) on document_images.

        Failures are logged at WARNING because the flag is the canonical
        O(1) lookup retrievers use — silent drift here means a row is in
        VECS but the flag is false, so the retriever skips it and the
        whole specialized embedding stops contributing to search.
        """
        try:
            self._get_supabase_rest().client.table('document_images').update(
                {flag_column: True}
            ).eq('id', image_id).execute()
        except Exception as flag_err:
            logger.warning(
                f"⚠️ Failed to set {flag_column} for image {image_id} — "
                f"VECS↔flag drift; row will be skipped by retrievers using flag pre-filter: {flag_err}"
            )

    def image_has_embedding(self, image_id: str, flag_column: str) -> bool:
        """O(1) presence check on document_images.

        Retrievers should call this before issuing a VECS query when the
        cost of a missing row is high (e.g. specialized-only searches),
        instead of round-tripping to the vector collection.
        """
        try:
            row = self._get_supabase_rest().client.table('document_images').select(
                flag_column
            ).eq('id', image_id).single().execute()
            return bool(row.data and row.data.get(flag_column))
        except Exception:
            return False

    def _clear_image_flags(self, image_id: str, flag_columns: List[str]) -> None:
        """Clear has_*_slig flags after a delete from VECS so the row is no
        longer surfaced by retrievers that pre-filter on the flags.
        """
        if not flag_columns:
            return
        try:
            self._get_supabase_rest().client.table('document_images').update(
                {col: False for col in flag_columns}
            ).eq('id', image_id).execute()
        except Exception as flag_err:
            logger.warning(
                f"⚠️ Failed to clear flags {flag_columns} for image {image_id}: {flag_err}"
            )


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

            # Update canonical presence flag on document_images
            self._set_image_flag(image_id, 'has_slig_embedding')

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

            # Update canonical presence flags on document_images
            for record in records:
                image_id = record[0] if record else None
                if image_id:
                    self._set_image_flag(image_id, 'has_slig_embedding')

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
                        dimension=768  # SLIG SigLIP2 cloud endpoint
                    )

                    # Upsert single record
                    collection.upsert(records=[(image_id, embeddings[embedding_type], metadata)])

                    # Update canonical presence flag (has_color_slig / has_texture_slig / etc.)
                    self._set_image_flag(image_id, f'has_{embedding_type}_slig')

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

            # Update canonical presence flag on document_images
            self._set_image_flag(image_id, 'has_understanding_embedding')

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

            # Search - Run in thread pool for true async parallelism.
            # include_value=True is required to get the distance back; without
            # it the unpack below would ValueError because vecs returns a
            # 2-tuple instead of a 3-tuple.
            import asyncio
            results = await asyncio.to_thread(
                collection.query,
                data=query_embedding,
                limit=limit,
                filters=filters,
                include_value=True,
                include_metadata=include_metadata
            )

            # Format results — handle (id, distance, metadata?) variants.
            formatted_results = []
            for result_tuple in results:
                if include_metadata:
                    image_id, distance, metadata = result_tuple
                else:
                    image_id, distance = result_tuple
                    metadata = None
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
            # Post-migration: all 4 specialized collections are 768D halfvec, matching
            # the SLIG cloud endpoint output. Any dimension mismatch here is a real
            # bug (e.g. caller passing the wrong-size query embedding) — log as ERROR.
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
            # Run all searches in parallel.
            tasks = [search_visual(), search_understanding()]

            # Specialized searches only run when a real specialized query
            # embedding is provided. Falling back to the visual embedding
            # silently degrades precision (the color collection is then
            # queried with a generic visual vector) — the caller should
            # explicitly opt out by omitting the type instead.
            specialized_types_to_run: List[str] = []
            for emb_type in ['color', 'texture', 'style', 'material']:
                query_emb = specialized_query_embeddings.get(emb_type)
                if query_emb is None:
                    continue
                specialized_types_to_run.append(emb_type)
                tasks.append(search_specialized(emb_type, query_emb))

            # Execute all searches in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)

            def _ok(idx: int) -> List[Dict[str, Any]]:
                if idx >= len(results) or isinstance(results[idx], Exception):
                    return []
                return results[idx]

            visual_results = _ok(0)
            understanding_results = _ok(1)

            # Specialized results indexed by their position in tasks list.
            specialized_results: Dict[str, List[Dict[str, Any]]] = {}
            for offset, emb_type in enumerate(specialized_types_to_run):
                specialized_results[emb_type] = _ok(2 + offset)

            color_results = specialized_results.get('color', [])
            texture_results = specialized_results.get('texture', [])
            style_results = specialized_results.get('style', [])
            material_results = specialized_results.get('material', [])

            # Build lookup maps for all scores
            understanding_scores = {r['image_id']: r['similarity_score'] for r in understanding_results}
            specialized_scores: Dict[str, Dict[str, float]] = {
                emb_type: {r['image_id']: r['similarity_score'] for r in rows}
                for emb_type, rows in specialized_results.items()
            }

            all_image_ids = set()
            for result in visual_results:
                all_image_ids.add(result['image_id'])
            all_image_ids.update(understanding_scores.keys())
            for score_map in specialized_scores.values():
                all_image_ids.update(score_map.keys())

            visual_scores_map = {r['image_id']: r for r in visual_results}

            has_understanding = bool(understanding_query_embedding and understanding_results)
            specialized_count = len(specialized_types_to_run)

            # Weights are normalized over the dimensions actually queried,
            # so omitting (say) color doesn't dilute the remaining vectors.
            base_weights: Dict[str, float] = {'visual': 0.25}
            if has_understanding:
                base_weights['understanding'] = 0.20
            if specialized_count > 0:
                # Distribute remaining mass across present specialized types.
                remaining = 1.0 - sum(base_weights.values())
                per_specialized = remaining / specialized_count
                for emb_type in specialized_types_to_run:
                    base_weights[emb_type] = per_specialized
            else:
                # Re-normalize so visual + understanding sum to 1.0.
                total = sum(base_weights.values()) or 1.0
                base_weights = {k: v / total for k, v in base_weights.items()}

            combined_results = []
            for image_id in all_image_ids:
                visual_data = visual_scores_map.get(image_id)
                visual_score = visual_data.get('similarity_score', 0.0) if visual_data else 0.0
                understanding_score = understanding_scores.get(image_id, 0.0)

                # Per-vector scores: 0.0 when this image was not in that collection.
                # No silent fallback to visual_score.
                per_type_scores = {
                    emb_type: specialized_scores[emb_type].get(image_id, 0.0)
                    for emb_type in specialized_types_to_run
                }

                combined_score = (
                    base_weights.get('visual', 0.0) * visual_score
                    + base_weights.get('understanding', 0.0) * understanding_score
                    + sum(base_weights.get(t, 0.0) * s for t, s in per_type_scores.items())
                )

                combined_result = {
                    'image_id': image_id,
                    'similarity_score': visual_score,
                    'combined_score': combined_score,
                    'scores': {
                        'visual': visual_score,
                        'understanding': understanding_score,
                        'color': per_type_scores.get('color', 0.0),
                        'texture': per_type_scores.get('texture', 0.0),
                        'style': per_type_scores.get('style', 0.0),
                        'material': per_type_scores.get('material', 0.0),
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
        """Delete an image's vectors from every collection and clear its flags.

        Returns True only if the primary delete succeeds. Specialized collection
        deletes are wrapped because not every image has every embedding type,
        but failures are logged so we can spot real problems.
        """
        try:
            collection = self.get_or_create_collection(
                name="image_slig_embeddings",
                dimension=768
            )
            collection.delete(ids=[image_id])

            for col_name, dim in [
                ("image_understanding_embeddings", 1024),
                ("image_color_embeddings", 768),
                ("image_texture_embeddings", 768),
                ("image_style_embeddings", 768),
                ("image_material_embeddings", 768),
            ]:
                try:
                    col = self.get_or_create_collection(name=col_name, dimension=dim)
                    col.delete(ids=[image_id])
                except Exception as col_err:
                    logger.warning(f"Specialized delete from {col_name} for {image_id} failed: {col_err}")

            self._clear_image_flags(image_id, [
                'has_slig_embedding',
                'has_understanding_embedding',
                'has_color_slig',
                'has_texture_slig',
                'has_style_slig',
                'has_material_slig',
            ])

            logger.debug(f"✅ Deleted embeddings + cleared flags for image {image_id}")
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

                for col_name, dim in [
                    ("image_understanding_embeddings", 1024),
                    ("image_color_embeddings", 768),
                    ("image_texture_embeddings", 768),
                    ("image_style_embeddings", 768),
                    ("image_material_embeddings", 768),
                ]:
                    try:
                        col = self.get_or_create_collection(name=col_name, dimension=dim)
                        col.delete(ids=image_ids)
                    except Exception as col_err:
                        logger.warning(f"Specialized delete from {col_name} for document {document_id} failed: {col_err}")

                # Clear all flags in one batch update per document
                try:
                    self._get_supabase_rest().client.table('document_images').update({
                        'has_slig_embedding': False,
                        'has_understanding_embedding': False,
                        'has_color_slig': False,
                        'has_texture_slig': False,
                        'has_style_slig': False,
                        'has_material_slig': False,
                    }).eq('document_id', document_id).execute()
                except Exception as flag_err:
                    logger.warning(f"Failed to clear flags for document {document_id}: {flag_err}")

                logger.info(f"✅ Deleted {len(image_ids)} embeddings + cleared flags for document {document_id}")
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


