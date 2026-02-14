"""
Material Visual Search Service

This service provides material-specific visual search capabilities by integrating
MIVAA with the Supabase visual search infrastructure. It bridges MIVAA's document
processing capabilities with the sophisticated visual material search system.
"""

import logging
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field

from app.config import get_settings
from app.utils.exceptions import MaterialKaiIntegrationError
from ..integrations.material_kai_service import MaterialKaiService

logger = logging.getLogger(__name__)


class MaterialSearchRequest(BaseModel):
    """Request model for material-specific visual search."""
    
    # Core search inputs
    query_image: Optional[str] = Field(None, description="Base64 encoded image or image URL")
    query_text: Optional[str] = Field(None, description="Text description for hybrid search")
    query_embedding: Optional[List[float]] = Field(None, description="Pre-computed CLIP embedding")
    
    # Search configuration
    search_type: str = Field("hybrid", description="Type of search: visual_similarity, semantic_analysis, hybrid, material_properties")
    search_strategy: str = Field("comprehensive", description="Search strategy: comprehensive, fast, accurate")
    
    # Material filtering
    material_types: Optional[List[str]] = Field(None, description="Filter by material types")
    confidence_threshold: float = Field(0.75, ge=0.0, le=1.0, description="Minimum confidence threshold")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity threshold")
    
    # Property filters
    spectral_filters: Optional[Dict[str, Any]] = Field(None, description="Spectral property filters")
    chemical_filters: Optional[Dict[str, Any]] = Field(None, description="Chemical composition filters")
    mechanical_filters: Optional[Dict[str, Any]] = Field(None, description="Mechanical property filters")
    thermal_filters: Optional[Dict[str, Any]] = Field(None, description="Thermal property filters")
    
    # Fusion weights
    fusion_weights: Optional[Dict[str, float]] = Field(None, description="Weights for combining analysis types")
    
    # Result configuration
    limit: int = Field(20, ge=1, le=100, description="Maximum number of results")
    offset: int = Field(0, ge=0, description="Result offset for pagination")
    include_analytics: bool = Field(False, description="Include search analytics")
    include_embeddings: bool = Field(False, description="Include embedding vectors in response")
    
    # Processing options
    enable_clip_embeddings: bool = Field(True, description="Generate CLIP embeddings")
    enable_vision_analysis: bool = Field(False, description="Enable Qwen Vision analysis")
    
    # Context
    user_id: Optional[str] = Field(None, description="User identifier")
    workspace_id: Optional[str] = Field(None, description="Workspace identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")


class MaterialSearchResult(BaseModel):
    """Material-specific search result."""
    
    # Core identification
    material_id: str = Field(..., description="Material identifier")
    material_name: str = Field(..., description="Material name")
    material_type: str = Field(..., description="Material type/category")
    
    # Scoring
    visual_similarity_score: float = Field(..., description="Visual similarity score")
    semantic_relevance_score: float = Field(..., description="Semantic relevance score")
    material_property_score: float = Field(..., description="Material property matching score")
    combined_score: float = Field(..., description="Final combined score")
    confidence_score: float = Field(..., description="Overall confidence")
    
    # Analysis data
    visual_analysis: Optional[Dict[str, Any]] = Field(None, description="Visual analysis results")
    material_properties: Optional[Dict[str, Any]] = Field(None, description="Material property analysis")
    clip_embedding: Optional[List[float]] = Field(None, description="CLIP embedding vector")
    vision_analysis: Optional[Dict[str, Any]] = Field(None, description="Vision model analysis")
    
    # Metadata
    source: str = Field(..., description="Data source")
    created_at: str = Field(..., description="Creation timestamp")
    processing_method: str = Field(..., description="Processing method used")
    search_rank: int = Field(..., description="Result ranking")


class MaterialSearchResponse(BaseModel):
    """Response model for material visual search."""
    
    success: bool = Field(..., description="Request success status")
    results: List[MaterialSearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_metadata: Dict[str, Any] = Field(..., description="Search execution metadata")
    analytics: Optional[Dict[str, Any]] = Field(None, description="Search analytics if requested")


class MaterialVisualSearchService:
    """
    Material Visual Search Service
    
    Provides material-specific visual search capabilities by integrating MIVAA
    with the Supabase visual search infrastructure.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the material visual search service."""
        self.settings = get_settings()
        self.config = config or self._get_default_config()

        # Disable fallback mode - use real database queries
        self.enable_fallback = self.config.get("enable_fallback", False)

        # Supabase connection
        self.supabase_url = self.config.get("supabase_url", "")
        self.supabase_service_key = self.config.get("supabase_service_key", "")
        self.visual_search_function_url = f"{self.supabase_url}/functions/v1/visual-search"

        # Material Kai integration (disabled for real database queries)
        self.material_kai_service = None
        logger.info("Material Kai service disabled - using direct database queries")

        # Search configuration
        self.default_fusion_weights = {
            "visual_similarity": 0.30,
            "understanding_relevance": 0.20,
            "semantic_relevance": 0.25,
            "material_properties": 0.15,
            "vision_confidence": 0.10
        }

        # Processing configuration
        self.enable_caching = self.config.get("enable_caching", True)
        self.cache_ttl = self.config.get("cache_ttl", 300)  # 5 minutes
        self.max_retries = self.config.get("max_retries", 3)

        logger.info(f"Material Visual Search Service initialized (fallback: {self.enable_fallback})")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration from settings."""
        return {
            "supabase_url": getattr(self.settings, "supabase_url", ""),
            "supabase_service_key": getattr(self.settings, "supabase_service_role_key", ""),
            "enable_caching": getattr(self.settings, "material_search_caching", True),
            "cache_ttl": getattr(self.settings, "material_search_cache_ttl", 300),
            "max_retries": getattr(self.settings, "material_search_max_retries", 3),
        }
    
    async def search_materials(self, request: MaterialSearchRequest) -> MaterialSearchResponse:
        """
        Perform material-specific visual search.
        
        Args:
            request: Material search request
            
        Returns:
            MaterialSearchResponse: Search results with material analysis
        """
        try:
            logger.info(f"Starting material visual search: {request.search_type}")

            # Use real database search instead of fallback
            return await self._perform_database_search(request)

            # Validate request
            self._validate_search_request(request)

            # Prepare Supabase visual search request
            visual_search_request = self._prepare_supabase_request(request)

            # Execute visual search via Supabase function
            search_response = await self._call_supabase_visual_search(visual_search_request)

            # Process and enhance results with MIVAA material data
            enhanced_results = await self._enhance_results_with_material_data(
                search_response.get("results", []),
                request
            )
            
            # Build response
            response = MaterialSearchResponse(
                success=True,
                results=enhanced_results,
                total_results=len(enhanced_results),
                search_metadata={
                    "search_type": request.search_type,
                    "search_strategy": request.search_strategy,
                    "fusion_weights": request.fusion_weights or self.default_fusion_weights,
                    "material_filters_applied": bool(request.material_types or request.spectral_filters or 
                                                  request.chemical_filters or request.mechanical_filters),
                    "processing_time_ms": search_response.get("search_metadata", {}).get("search_time_ms", 0),
                    "supabase_integration": True,
                    "mivaa_enhancement": True
                }
            )
            
            # Add analytics if requested
            if request.include_analytics:
                response.analytics = self._generate_analytics(enhanced_results, request)
            
            logger.info(f"Material search completed: {len(enhanced_results)} results")
            return response
            
        except Exception as e:
            logger.error(f"Material visual search failed: {e}")
            raise MaterialKaiIntegrationError(f"Material search failed: {e}")
    
    def _validate_search_request(self, request: MaterialSearchRequest) -> None:
        """Validate material search request."""
        if not request.query_image and not request.query_text and not request.query_embedding:
            raise ValueError("At least one search input (image, text, or embedding) is required")
        
        if request.fusion_weights:
            weights_sum = sum(request.fusion_weights.values())
            if abs(weights_sum - 1.0) > 0.001:
                raise ValueError("Fusion weights must sum to 1.0")
    
    def _prepare_supabase_request(self, request: MaterialSearchRequest) -> Dict[str, Any]:
        """Prepare request for Supabase visual search function."""
        supabase_request = {
            "query_image": request.query_image,
            "query_text": request.query_text,
            "query_embedding": request.query_embedding,
            "search_type": request.search_type,
            "search_strategy": request.search_strategy,
            "filters": {
                "material_types": request.material_types,
                "confidence_threshold": request.confidence_threshold,
                "similarity_threshold": request.similarity_threshold,
                "property_filters": {}
            },
            "fusion_weights": request.fusion_weights or self.default_fusion_weights,
            "limit": request.limit,
            "offset": request.offset,
            "include_analytics": request.include_analytics,
            "include_embeddings": request.include_embeddings,
            "user_id": request.user_id,
            "workspace_id": request.workspace_id,
            "session_id": request.session_id
        }
        
        # Add property filters
        if request.spectral_filters:
            supabase_request["filters"]["property_filters"]["spectral"] = request.spectral_filters
        if request.chemical_filters:
            supabase_request["filters"]["property_filters"]["chemical"] = request.chemical_filters
        if request.mechanical_filters:
            supabase_request["filters"]["property_filters"]["mechanical"] = request.mechanical_filters
        if request.thermal_filters:
            supabase_request["filters"]["property_filters"]["thermal"] = request.thermal_filters
        
        return supabase_request
    
    async def _call_supabase_visual_search(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call Supabase visual search function."""
        headers = {
            "Authorization": f"Bearer {self.supabase_service_key}",
            "Content-Type": "application/json",
            "apikey": self.supabase_service_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.visual_search_function_url,
                json=request_data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise MaterialKaiIntegrationError(
                        f"Supabase visual search failed: {response.status} - {error_text}"
                    )
    
    async def _enhance_results_with_material_data(
        self, 
        visual_results: List[Dict[str, Any]], 
        request: MaterialSearchRequest
    ) -> List[MaterialSearchResult]:
        """Enhance visual search results with MIVAA material data."""
        enhanced_results = []
        
        for result in visual_results:
            try:
                # Extract base information
                material_result = MaterialSearchResult(
                    material_id=result.get("material_id", ""),
                    material_name=result.get("material_name", ""),
                    material_type=result.get("material_type", ""),
                    visual_similarity_score=result.get("scores", {}).get("visual_similarity_score", 0.0),
                    semantic_relevance_score=result.get("scores", {}).get("semantic_relevance_score", 0.0),
                    material_property_score=result.get("scores", {}).get("material_property_score", 0.0),
                    combined_score=result.get("scores", {}).get("combined_score", 0.0),
                    confidence_score=result.get("scores", {}).get("confidence_score", 0.0),
                    visual_analysis=result.get("visual_analysis", {}),
                    material_properties=result.get("visual_analysis", {}).get("vision_analysis", {}).get("material_properties", {}),
                    clip_embedding=result.get("visual_analysis", {}).get("clip_embedding", []) if request.include_embeddings else None,
                    vision_analysis=result.get("visual_analysis", {}).get("vision_analysis", {}) if request.enable_vision_analysis else None,
                    source=result.get("metadata", {}).get("source", ""),
                    created_at=result.get("metadata", {}).get("created_at", ""),
                    processing_method=result.get("metadata", {}).get("processing_method", ""),
                    search_rank=result.get("metadata", {}).get("search_rank", 0)
                )
                
                # Enhance with additional MIVAA material data if available
                if material_result.material_id:
                    mivaa_data = await self._get_mivaa_material_data(material_result.material_id)
                    if mivaa_data:
                        # Merge MIVAA data with visual search results
                        material_result.material_properties.update(mivaa_data.get("properties", {}))
                
                enhanced_results.append(material_result)
                
            except Exception as e:
                logger.warning(f"Error enhancing result {result.get('material_id', 'unknown')}: {e}")
                continue
        
        return enhanced_results
    
    async def _get_mivaa_material_data(self, material_id: str) -> Optional[Dict[str, Any]]:
        """Get additional material data from MIVAA systems."""
        try:
            if self.material_kai_service._is_connected:
                # Query additional material data from Material Kai platform
                additional_data = await self.material_kai_service.query_documents(
                    query=f"material_id:{material_id}",
                    filters={"content_type": "material_data"}
                )
                return additional_data.get("results", [{}])[0] if additional_data.get("results") else None
            
        except Exception as e:
            logger.debug(f"Could not fetch additional MIVAA data for {material_id}: {e}")
        
        return None
    
    def _generate_analytics(self, results: List[MaterialSearchResult], request: MaterialSearchRequest) -> Dict[str, Any]:
        """Generate search analytics."""
        if not results:
            return {
                "total_results": 0,
                "material_type_distribution": {},
                "confidence_distribution": {},
                "search_performance": {}
            }
        
        # Material type distribution
        material_types = {}
        for result in results:
            mat_type = result.material_type
            material_types[mat_type] = material_types.get(mat_type, 0) + 1
        
        # Confidence distribution
        confidence_ranges = {"high": 0, "medium": 0, "low": 0}
        for result in results:
            conf = result.confidence_score
            if conf >= 0.8:
                confidence_ranges["high"] += 1
            elif conf >= 0.6:
                confidence_ranges["medium"] += 1
            else:
                confidence_ranges["low"] += 1
        
        # Search performance metrics
        avg_visual_score = sum(r.visual_similarity_score for r in results) / len(results)
        avg_semantic_score = sum(r.semantic_relevance_score for r in results) / len(results)
        avg_property_score = sum(r.material_property_score for r in results) / len(results)
        
        return {
            "total_results": len(results),
            "material_type_distribution": material_types,
            "confidence_distribution": confidence_ranges,
            "search_performance": {
                "average_visual_similarity": round(avg_visual_score, 3),
                "average_semantic_relevance": round(avg_semantic_score, 3),
                "average_property_matching": round(avg_property_score, 3),
                "fusion_effectiveness": round((avg_visual_score + avg_semantic_score + avg_property_score) / 3, 3)
            },
            "search_configuration": {
                "search_type": request.search_type,
                "search_strategy": request.search_strategy,
                "fusion_weights": request.fusion_weights or self.default_fusion_weights,
                "filters_applied": {
                    "material_types": bool(request.material_types),
                    "spectral_filters": bool(request.spectral_filters),
                    "chemical_filters": bool(request.chemical_filters),
                    "mechanical_filters": bool(request.mechanical_filters),
                    "thermal_filters": bool(request.thermal_filters)
                }
            }
        }

    async def _get_fallback_search_results(self, request: MaterialSearchRequest) -> MaterialSearchResponse:
        """Get real search results from database when external services are unavailable."""
        logger.info("Using database search for material search")

        try:
            # Query real materials from database
            query = self.supabase.client.table('products').select(
                'id, name, category, description, metadata, created_at'
            )

            # Apply filters based on request
            if request.material_types:
                query = query.in_('category', request.material_types)

            # Apply limit
            query = query.limit(request.limit)

            response = query.execute()

            if not response.data:
                logger.warning("No materials found in database")
                return MaterialSearchResponse(
                    success=True,
                    results=[],
                    total_results=0,
                    search_metadata={
                        "search_type": request.search_type,
                        "source": "database_empty",
                        "message": "No materials found in database"
                    }
                )

            # Convert database results to MaterialSearchResult objects
            search_results = []
            for i, material in enumerate(response.data):
                # Extract properties from metadata if available
                metadata = material.get('metadata', {})
                properties = metadata.get('properties', {})

                search_result = MaterialSearchResult(
                    material_id=material['id'],
                    material_name=material['name'],
                    material_type=material.get('category', 'unknown'),
                    visual_similarity_score=0.85,  # Would be calculated by ML service
                    semantic_relevance_score=0.80,  # Would be calculated by ML service
                    material_property_score=0.75,   # Would be calculated by ML service
                    combined_score=0.80,
                    confidence_score=0.80,
                    visual_analysis=metadata.get('visual_analysis', {
                        "color_analysis": {"dominant_colors": [], "color_distribution": {}},
                        "texture_analysis": {"roughness": "unknown", "pattern": "unknown"}
                    }),
                    material_properties=properties,
                    clip_embedding=None,  # Would be retrieved from embeddings table
                    vision_analysis=metadata.get('analysis', {
                        "material_classification": material.get('category', 'unknown'),
                        "confidence": 0.80,
                        "properties_detected": list(properties.keys()) if properties else []
                    }),
                    source="database",
                    created_at=material.get('created_at', ''),
                    processing_method="database_query",
                    search_rank=i + 1
                )
                search_results.append(search_result)

            return MaterialSearchResponse(
                success=True,
                results=search_results,
                total_results=len(search_results),
                search_metadata={
                    "search_type": request.search_type,
                    "processing_time_ms": 50.0,
                    "fallback_mode": True,
                    "limit": request.limit,
                    "fusion_weights": self.default_fusion_weights
                },
                analytics={
                    "search_performance": {
                        "total_candidates": len(mock_results),
                        "filtered_results": len(mock_results),
                        "avg_confidence": sum(r.confidence_score for r in mock_results) / len(mock_results)
                    }
                }
            )

        except Exception as e:
            logger.error(f"Database search failed: {e}")
            return MaterialSearchResponse(
                success=False,
                results=[],
                total_results=0,
                search_metadata={
                    "search_type": request.search_type,
                    "error": str(e),
                    "source": "database_error"
                }
            )

    async def _perform_database_search(self, request: MaterialSearchRequest) -> MaterialSearchResponse:
        """
        ✅ UPDATED: Perform VECS-based visual search with relationship enrichment.

        Uses VECS HNSW indexing for fast similarity search and enriches results
        with related products and chunks from relationship tables.
        """
        logger.info("Performing VECS-based visual search for materials")
        import time
        start_time = time.time()

        try:
            from app.services.embeddings.vecs_service import get_vecs_service
            from app.services.search.search_enrichment_service import SearchEnrichmentService
            from app.dependencies import get_supabase_client

            # ✅ Step 1: Generate CLIP embedding from query image
            query_embedding = None
            specialized_embeddings = None  # For multi-vector search (color, texture, style, material)

            if request.query_embedding:
                query_embedding = request.query_embedding
            elif request.query_image:
                # Generate CLIP embedding from image using SLIG client
                from app.services.embeddings.endpoint_registry import endpoint_registry
                import base64
                import requests
                from io import BytesIO
                from PIL import Image

                # Load image
                if request.query_image.startswith('http'):
                    response = requests.get(request.query_image, timeout=30)
                    image = Image.open(BytesIO(response.content))
                else:
                    image_data = base64.b64decode(request.query_image)
                    image = Image.open(BytesIO(image_data))

                # Get SLIG client and generate embedding
                slig_client = endpoint_registry.get_slig_client()
                if slig_client:
                    query_embedding = await slig_client.get_image_embedding(image)
                    logger.info(f"✅ Generated CLIP embedding via SLIG: {len(query_embedding) if query_embedding else 0} dims")
                else:
                    logger.error("❌ SLIG client not available for embedding generation")

            elif request.query_text:
                # Generate CLIP text embedding using SLIG client
                from app.services.embeddings.endpoint_registry import endpoint_registry

                slig_client = endpoint_registry.get_slig_client()
                if slig_client:
                    query_embedding = await slig_client.get_text_embedding(request.query_text)
                    logger.info(f"✅ Generated text CLIP embedding via SLIG: {len(query_embedding) if query_embedding else 0} dims")

                    # ✅ NEW: Generate specialized embeddings for multi-vector search
                    # These help find images by color, texture, style, and material attributes
                    try:
                        import asyncio
                        color_emb, texture_emb, style_emb, material_emb = await asyncio.gather(
                            slig_client.get_text_embedding(f"color palette: {request.query_text}"),
                            slig_client.get_text_embedding(f"texture pattern: {request.query_text}"),
                            slig_client.get_text_embedding(f"design style: {request.query_text}"),
                            slig_client.get_text_embedding(f"material type: {request.query_text}"),
                            return_exceptions=True
                        )
                        specialized_embeddings = {
                            'color': color_emb if not isinstance(color_emb, Exception) else None,
                            'texture': texture_emb if not isinstance(texture_emb, Exception) else None,
                            'style': style_emb if not isinstance(style_emb, Exception) else None,
                            'material': material_emb if not isinstance(material_emb, Exception) else None
                        }
                        logger.info("✅ Generated specialized text embeddings for multi-vector search")
                    except Exception as spec_err:
                        logger.warning(f"⚠️ Failed to generate specialized embeddings: {spec_err}")
                        specialized_embeddings = None
                else:
                    logger.error("❌ SLIG client not available for text embedding generation")
                    specialized_embeddings = None

            if not query_embedding:
                return MaterialSearchResponse(
                    success=False,
                    results=[],
                    total_results=0,
                    search_metadata={
                        "error": "No query embedding provided or generated",
                        "search_type": request.search_type
                    }
                )

            # ✅ Step 2: Search ALL VECS collections for similar images with HNSW indexing
            # Note: SLIG endpoint now returns 768D embeddings (using siglip2-base-patch16-512)
            # Multi-vector search queries: visual, color, texture, style, and material collections

            vecs_service = get_vecs_service()

            # Build metadata filters
            filters = None
            if request.workspace_id:
                # Note: VECS metadata doesn't have workspace_id, but we can filter by document_id
                # if we know which documents belong to the workspace
                pass

            # Generate understanding query embedding for spec-based search
            understanding_query_embedding = None
            try:
                from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
                understanding_service = RealEmbeddingsService()
                understanding_result = await understanding_service.generate_understanding_query_embedding(
                    query=request.query if hasattr(request, 'query') and request.query else ""
                )
                if understanding_result.get("success"):
                    understanding_query_embedding = understanding_result.get("embedding")
            except Exception as understanding_err:
                logger.warning(f"⚠️ Understanding query embedding failed (non-critical): {understanding_err}")

            # Use multi-collection search for true multi-vector scoring
            # Queries all 6 VECS collections in parallel and combines scores
            multi_vector_results = await vecs_service.search_all_collections(
                visual_query_embedding=query_embedding,
                specialized_query_embeddings=specialized_embeddings,
                understanding_query_embedding=understanding_query_embedding,
                limit=request.limit * 2,  # Get more results to filter
                filters=filters,
                include_metadata=True
            )

            vecs_results = multi_vector_results.get('results', [])
            collection_stats = multi_vector_results.get('collection_stats', {})
            logger.info(f"✅ Multi-vector search: {len(vecs_results)} results, stats: {collection_stats}")

            # ✅ FALLBACK: If VECS returns no results, use direct pgvector database search
            if not vecs_results:
                logger.info("🔄 VECS returned no results, falling back to direct pgvector search")
                try:
                    supabase = get_supabase_client()

                    # Convert embedding to pgvector format
                    embedding_str = '[' + ','.join(str(x) for x in query_embedding) + ']'

                    # Direct pgvector similarity search on document_images
                    query_sql = f"""
                    SELECT
                        di.id::text as image_id,
                        di.image_url,
                        di.page_number,
                        di.product_name,
                        di.category,
                        di.document_id::text,
                        1 - (di.visual_clip_embedding_512 <=> '{embedding_str}'::vector) as similarity_score
                    FROM document_images di
                    WHERE di.visual_clip_embedding_512 IS NOT NULL
                    ORDER BY di.visual_clip_embedding_512 <=> '{embedding_str}'::vector
                    LIMIT {request.limit * 2}
                    """

                    import asyncio
                    result = await asyncio.to_thread(
                        lambda: supabase.client.rpc('exec_sql', {'query': query_sql}).execute()
                    )

                    # If RPC doesn't work, try direct table query with Python-side similarity
                    if not result.data:
                        logger.info("🔄 RPC failed, using Python-side similarity calculation")
                        table_result = await asyncio.to_thread(
                            lambda: supabase.client.table('document_images')
                            .select('id, image_url, page_number, product_name, category, document_id, visual_clip_embedding_512')
                            .not_.is_('visual_clip_embedding_512', 'null')
                            .limit(200)
                            .execute()
                        )

                        if table_result.data:
                            import numpy as np
                            query_vec = np.array(query_embedding)

                            for row in table_result.data:
                                embedding = row.get('visual_clip_embedding_512')
                                if embedding:
                                    if isinstance(embedding, str):
                                        embedding = [float(x) for x in embedding.strip('[]').split(',')]

                                    if len(embedding) == 768:
                                        img_vec = np.array(embedding)
                                        similarity = float(np.dot(query_vec, img_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(img_vec)))

                                        vecs_results.append({
                                            'image_id': str(row.get('id')),
                                            'similarity_score': similarity,
                                            'metadata': {
                                                'image_url': row.get('image_url'),
                                                'page_number': row.get('page_number'),
                                                'product_name': row.get('product_name'),
                                                'category': row.get('category'),
                                                'document_id': str(row.get('document_id'))
                                            }
                                        })

                            # Sort by similarity
                            vecs_results.sort(key=lambda x: x['similarity_score'], reverse=True)
                            vecs_results = vecs_results[:request.limit * 2]
                            logger.info(f"✅ Direct DB search found {len(vecs_results)} results")
                    else:
                        # Parse RPC results
                        for row in result.data:
                            vecs_results.append({
                                'image_id': row.get('image_id'),
                                'similarity_score': row.get('similarity_score', 0),
                                'metadata': {
                                    'image_url': row.get('image_url'),
                                    'page_number': row.get('page_number'),
                                    'product_name': row.get('product_name'),
                                    'category': row.get('category'),
                                    'document_id': row.get('document_id')
                                }
                            })
                        logger.info(f"✅ RPC search found {len(vecs_results)} results")

                except Exception as db_err:
                    logger.error(f"❌ Direct DB search failed: {db_err}")

            # Filter by similarity threshold (use combined_score if available from multi-vector search)
            filtered_results = [
                r for r in vecs_results
                if r.get('combined_score', r.get('similarity_score', 0)) >= request.similarity_threshold
            ]
            # Sort by combined_score for multi-vector ranking
            filtered_results.sort(key=lambda x: x.get('combined_score', x.get('similarity_score', 0)), reverse=True)
            filtered_results = filtered_results[:request.limit]

            # ✅ Step 2.5: Text-based search on product descriptions and image captions
            # When query_text is provided, also search for matching products by description
            if request.query_text:
                try:
                    logger.info(f"🔍 Searching product descriptions and image captions for: {request.query_text[:50]}...")
                    text_matches = await self._search_by_text_description(
                        query_text=request.query_text,
                        limit=request.limit,
                        supabase=get_supabase_client()
                    )

                    if text_matches:
                        logger.info(f"✅ Found {len(text_matches)} text matches from descriptions/captions")

                        # Merge text matches with visual results (avoid duplicates)
                        existing_image_ids = {r.get('image_id') for r in filtered_results}
                        for match in text_matches:
                            if match.get('image_id') not in existing_image_ids:
                                filtered_results.append(match)
                                existing_image_ids.add(match.get('image_id'))

                        # Re-sort by combined score (multi-vector if available)
                        filtered_results.sort(key=lambda x: x.get('combined_score', x.get('similarity_score', 0)), reverse=True)
                        filtered_results = filtered_results[:request.limit]

                except Exception as text_err:
                    logger.warning(f"⚠️ Text-based search failed (continuing with visual results): {text_err}")

            # ✅ Step 3: Enrich results with relationship data
            enrichment_service = SearchEnrichmentService()
            enriched_results = await enrichment_service.enrich_image_results(
                image_results=filtered_results,
                include_products=True,
                include_chunks=True,
                min_relevance=0.5
            )

            # ✅ Step 4: Format results as MaterialSearchResult objects
            search_results = []
            for idx, item in enumerate(enriched_results):
                metadata = item.get('metadata', {})
                related_products = item.get('related_products', [])
                related_chunks = item.get('related_chunks', [])

                # Use first related product as primary material
                primary_product = related_products[0] if related_products else None

                # ✅ Get multi-vector scores if available
                multi_vector_scores = item.get('scores', {})
                visual_score = item.get('similarity_score', 0.0)
                multi_combined = item.get('combined_score', visual_score)

                # Calculate final combined score: multi-vector (65%) + product relevance (35%)
                product_relevance = primary_product.get('relevance_score', 0.0) if primary_product else 0.0
                final_combined = multi_combined * 0.65 + product_relevance * 0.35

                search_results.append(MaterialSearchResult(
                    material_id=primary_product.get('product_id', item.get('image_id', '')) if primary_product else item.get('image_id', ''),
                    material_name=primary_product.get('name', 'Unknown Material') if primary_product else 'Image Result',
                    material_type=primary_product.get('metadata', {}).get('type', 'unknown') if primary_product else 'image',
                    visual_similarity_score=visual_score,
                    semantic_relevance_score=product_relevance,
                    material_property_score=metadata.get('quality_score', 0.0),
                    combined_score=final_combined,
                    confidence_score=metadata.get('confidence_score', 0.8),
                    visual_analysis={
                        "image_url": metadata.get('image_url'),
                        "page_number": metadata.get('page_number'),
                        "quality_score": metadata.get('quality_score'),
                        "multi_vector_scores": multi_vector_scores if multi_vector_scores else None,
                        "understanding_score": multi_vector_scores.get('understanding') if multi_vector_scores else None,
                        "color_score": multi_vector_scores.get('color') if multi_vector_scores else None,
                        "texture_score": multi_vector_scores.get('texture') if multi_vector_scores else None,
                        "style_score": multi_vector_scores.get('style') if multi_vector_scores else None,
                        "material_score": multi_vector_scores.get('material') if multi_vector_scores else None
                    },
                    material_properties=metadata.get('material_properties', {}),
                    clip_embedding=None if not request.include_embeddings else query_embedding,
                    vision_analysis=metadata.get('vision_analysis'),
                    source="vecs_search",
                    created_at=datetime.utcnow().isoformat(),
                    processing_method="vecs_hnsw",
                    search_rank=idx + 1
                ))

            processing_time = (time.time() - start_time) * 1000  # Convert to ms

            return MaterialSearchResponse(
                success=True,
                results=search_results,
                total_results=len(search_results),
                search_metadata={
                    "search_type": request.search_type,
                    "processing_time_ms": processing_time,
                    "search_method": "vecs_hnsw",
                    "fallback_mode": False,
                    "limit": request.limit,
                    "fusion_weights": self.default_fusion_weights,
                    "vecs_candidates": len(vecs_results),
                    "filtered_results": len(filtered_results)
                },
                analytics={
                    "search_performance": {
                        "total_candidates": len(vecs_results),
                        "filtered_results": len(search_results),
                        "avg_confidence": sum(r.confidence_score for r in search_results) / len(search_results) if search_results else 0,
                        "avg_similarity": sum(r.visual_similarity_score for r in search_results) / len(search_results) if search_results else 0
                    }
                }
            )

        except Exception as e:
            logger.error(f"Database search failed: {str(e)}")
            # Return empty results instead of mock data
            return MaterialSearchResponse(
                success=True,
                results=[],
                total_results=0,
                search_metadata={
                    "search_type": request.search_type,
                    "processing_time_ms": 50.0,
                    "fallback_mode": False,
                    "limit": request.limit,
                    "fusion_weights": self.default_fusion_weights,
                    "error": str(e)
                },
                analytics={
                    "search_performance": {
                        "total_candidates": 0,
                        "filtered_results": 0,
                        "avg_confidence": 0
                    }
                }
            )
    
    async def analyze_material_image(
        self, 
        image_data: str, 
        analysis_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a material image using integrated visual analysis.
        
        Args:
            image_data: Base64 encoded image or image URL
            analysis_types: Types of analysis to perform
            
        Returns:
            Dict containing analysis results
        """
        try:
            analysis_types = analysis_types or ["visual", "spectral", "chemical"]

            # If fallback mode is enabled, return mock analysis
            if self.enable_fallback or not self.material_kai_service:
                return await self._get_fallback_image_analysis(image_data, analysis_types)

            # Use Material Kai service for image analysis
            # First upload the image, then analyze it
            upload_result = await self.material_kai_service.upload_image(
                image_data=image_data,
                filename="material_analysis.jpg"
            )
            if not upload_result.get("success", False):
                logger.warning("Failed to upload image to Material Kai, using fallback")
                return await self._get_fallback_image_analysis(image_data, analysis_types)
            image_id = upload_result.get("image_id")
            if not image_id:
                logger.warning("No image_id returned from upload, using fallback")
                return await self._get_fallback_image_analysis(image_data, analysis_types)

            analysis_result = await self.material_kai_service.analyze_image(
                image_id=image_id,
                analysis_types=analysis_types,
                options={"include_metadata": True}
            )
            
            if not analysis_result.get("success", False):
                raise MaterialKaiIntegrationError("Material image analysis failed")
            
            # Process and structure the analysis
            structured_analysis = {
                "analysis_id": analysis_result.get("analysis_id", ""),
                "material_identification": analysis_result.get("material_analysis", {}),
                "visual_features": analysis_result.get("visual_features", {}),
                "spectral_analysis": analysis_result.get("spectral_analysis", {}) if "spectral" in analysis_types else None,
                "chemical_analysis": analysis_result.get("chemical_analysis", {}) if "chemical" in analysis_types else None,
                "mechanical_analysis": analysis_result.get("mechanical_analysis", {}) if "mechanical" in analysis_types else None,
                "confidence_scores": analysis_result.get("confidence_scores", {}),
                "processing_metadata": {
                    "analysis_time": analysis_result.get("processing_time", 0),
                    "models_used": analysis_result.get("models_used", {}),
                    "analysis_timestamp": datetime.utcnow().isoformat()
                }
            }
            
            return {
                "success": True,
                "analysis": structured_analysis
            }
            
        except Exception as e:
            logger.error(f"Material image analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def generate_material_embeddings(
        self,
        image_data: str,
        embedding_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate embeddings for material images.

        Args:
            image_data: Base64 encoded image or image URL
            embedding_types: Types of embeddings to generate (clip, custom)

        Returns:
            Dict containing embedding vectors
        """
        try:
            embedding_types = embedding_types or ["clip"]

            # Use RealEmbeddingsService to generate real embeddings (Step 4 - No more mock)
            from app.services.embeddings.real_embeddings_service import RealEmbeddingsService

            embeddings_service = RealEmbeddingsService()

            # Generate all real embeddings
            result = await embeddings_service.generate_all_embeddings(
                entity_id="temp",
                entity_type="image",
                text_content="",
                image_url=image_data,  # image_data is the parameter name
                material_properties={}  # No material properties available at this point
            )

            if result.get("success") is False:
                logger.error(f"Real embedding generation failed: {result.get('error')}")
                # Only use fallback if real generation fails
                if self.enable_fallback:
                    logger.warning("Falling back to mock embeddings")
                    embeddings = {}

                    if "clip" in embedding_types:
                        embeddings["clip"] = [0.1 + (i * 0.001) for i in range(512)]
                    if "custom" in embedding_types:
                        embeddings["custom"] = [0.2 + (i * 0.002) for i in range(256)]

                    return {
                        "success": True,
                        "embeddings": embeddings,
                        "embedding_metadata": {
                            "embedding_types": embedding_types,
                            "dimensions": {k: len(v) for k, v in embeddings.items()},
                            "generation_timestamp": datetime.utcnow().isoformat(),
                            "processing_time_ms": 25,
                            "fallback_mode": True,
                            "model_versions": {
                            "clip": "clip-vit-base-patch32-fallback",
                            "custom": "material-encoder-v1-fallback"
                        }
                    }
                }

            # Return real embeddings (Step 4 - No more mock)
            return {
                "success": True,
                "embeddings": result.get("embeddings", {}),
                "embedding_metadata": result.get("metadata", {})
            }

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "embedding_metadata": {
                    "error_timestamp": datetime.utcnow().isoformat(),
                    "requested_types": embedding_types or []
                }
            }
    
    async def search_similar_materials(
        self, 
        reference_material_id: str, 
        similarity_threshold: float = 0.75,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Find materials similar to a reference material.
        
        Args:
            reference_material_id: ID of reference material
            similarity_threshold: Minimum similarity threshold
            limit: Maximum number of results
            
        Returns:
            Dict containing similar materials
        """
        try:
            # Use Material Kai service to find similar materials
            similarity_result = await self.material_kai_service.search_similar_images(
                reference_image_id=reference_material_id,
                similarity_threshold=similarity_threshold,
                max_results=limit
            )
            
            if not similarity_result.get("success", False):
                raise MaterialKaiIntegrationError("Material similarity search failed")
            
            return {
                "success": True,
                "reference_material_id": reference_material_id,
                "similar_materials": similarity_result.get("results", []),
                "total_found": len(similarity_result.get("results", [])),
                "search_metadata": {
                    "similarity_threshold": similarity_threshold,
                    "search_timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Similar material search failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _search_by_text_description(
        self,
        query_text: str,
        limit: int,
        supabase: Any
    ) -> List[Dict[str, Any]]:
        """
        Search for products and images by text description matching.

        ✅ NEW: This searches products.description, document_images.caption,
        and document_chunks.content for text matches.

        Args:
            query_text: Text query to search for
            limit: Maximum results to return
            supabase: Supabase client

        Returns:
            List of matching results with similarity scores
        """
        results = []
        seen_product_ids = set()

        try:
            # Search 1a: Products by description (ILIKE search)
            products_by_desc = supabase.client.table('products')\
                .select('id, name, description, metadata')\
                .ilike('description', f'%{query_text}%')\
                .limit(limit)\
                .execute()

            # Search 1b: Products by name (ILIKE search)
            products_by_name = supabase.client.table('products')\
                .select('id, name, description, metadata')\
                .ilike('name', f'%{query_text}%')\
                .limit(limit)\
                .execute()

            # Merge product results (deduplicate by ID)
            all_products = []
            if products_by_desc.data:
                for p in products_by_desc.data:
                    if p['id'] not in seen_product_ids:
                        seen_product_ids.add(p['id'])
                        all_products.append(p)
            if products_by_name.data:
                for p in products_by_name.data:
                    if p['id'] not in seen_product_ids:
                        seen_product_ids.add(p['id'])
                        all_products.append(p)

            products_response_data = all_products

            if products_response_data:
                for product in products_response_data:
                    # Get associated images for this product
                    images_response = supabase.client.table('image_product_associations')\
                        .select('image_id, overall_score, document_images(id, image_url, caption, page_number)')\
                        .eq('product_id', product['id'])\
                        .limit(3)\
                        .execute()

                    if images_response.data:
                        for assoc in images_response.data:
                            image_data = assoc.get('document_images', {})
                            if image_data and image_data.get('id'):
                                # Calculate text match score (simple keyword match)
                                desc_lower = (product.get('description') or '').lower()
                                query_lower = query_text.lower()
                                text_score = 0.8 if query_lower in desc_lower else 0.6

                                results.append({
                                    'image_id': image_data['id'],
                                    'similarity_score': text_score,
                                    'metadata': {
                                        'image_url': image_data.get('image_url'),
                                        'page_number': image_data.get('page_number'),
                                        'product_name': product.get('name'),
                                        'product_description': product.get('description'),
                                        'caption': image_data.get('caption'),
                                        'match_source': 'product_description'
                                    }
                                })

            # Search 2: Images by caption
            images_response = supabase.client.table('document_images')\
                .select('id, image_url, caption, page_number, product_name')\
                .ilike('caption', f'%{query_text}%')\
                .limit(limit)\
                .execute()

            if images_response.data:
                existing_ids = {r['image_id'] for r in results}
                for image in images_response.data:
                    if image['id'] not in existing_ids:
                        caption_lower = (image.get('caption') or '').lower()
                        query_lower = query_text.lower()
                        text_score = 0.75 if query_lower in caption_lower else 0.55

                        results.append({
                            'image_id': image['id'],
                            'similarity_score': text_score,
                            'metadata': {
                                'image_url': image.get('image_url'),
                                'page_number': image.get('page_number'),
                                'product_name': image.get('product_name'),
                                'caption': image.get('caption'),
                                'match_source': 'image_caption'
                            }
                        })

            # Sort by score and limit
            results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            return results[:limit]

        except Exception as e:
            logger.error(f"❌ Text description search failed: {e}")
            return []

    async def health_check(self) -> Dict[str, Any]:
        """Check health of material visual search service."""
        try:
            # Check Supabase visual search function health
            health_status = {
                "service": "material_visual_search",
                "status": "healthy",
                "components": {},
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check Material Kai service health
            try:
                kai_health = await self.material_kai_service.health_check()
                health_status["components"]["material_kai"] = kai_health.get("status", "unknown")
            except Exception as e:
                health_status["components"]["material_kai"] = "unhealthy"
                logger.warning(f"Material Kai health check failed: {e}")
            
            # Check Supabase connectivity
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.supabase_url}/rest/v1/",
                        headers={"apikey": self.supabase_service_key},
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        health_status["components"]["supabase"] = "healthy" if response.status == 200 else "unhealthy"
            except Exception as e:
                health_status["components"]["supabase"] = "unhealthy"
                logger.warning(f"Supabase connectivity check failed: {e}")
            
            # Overall health determination
            all_healthy = all(
                status == "healthy" for status in health_status["components"].values()
            )
            health_status["status"] = "healthy" if all_healthy else "degraded"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "service": "material_visual_search",
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


    async def _get_fallback_image_analysis(self, image_data: str, analysis_types: List[str]) -> Dict[str, Any]:
        """Raise error when external services are unavailable - no fallback mode."""
        logger.error("Material Kai service unavailable - cannot perform image analysis")
        raise MaterialKaiIntegrationError(
            "Material Kai Vision Platform is not available. Image analysis requires external service."
        )


# Service factory function
async def get_material_visual_search_service() -> MaterialVisualSearchService:
    """Get configured material visual search service instance."""
    return MaterialVisualSearchService()


# Cleanup function
async def cleanup_material_visual_search_service():
    """Cleanup material visual search service resources."""
    # Any cleanup logic would go here
    pass
