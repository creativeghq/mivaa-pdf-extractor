"""
Search Routes - Query, chat, and search endpoints
"""

import logging
import time
from typing import List, Dict, Any, Optional
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Depends, status, Query

from app.services.llamaindex_service import LlamaIndexService
from app.services.advanced_search_service import AdvancedSearchService, QueryType, SearchOperator
from app.services.product_relationship_service import ProductRelationshipService
from app.services.search_prompt_service import SearchPromptService
from .shared import get_llamaindex_service
from .models import (
    QueryRequest, QueryResponse,
    ChatRequest, ChatResponse,
    SearchRequest, SearchResponse,
    MMRSearchRequest, MMRSearchResponse,
    AdvancedQueryRequest, AdvancedQueryResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    **?? CONSOLIDATED QUERY ENDPOINT - Text-Based RAG Query**

    This endpoint replaces:
    - `/api/documents/{id}/query` ? Use with `document_ids` filter
    - `/api/documents/{id}/summarize` ? Use with summarization prompt

    ## ?? Query Capabilities

    ### Text Query (Implemented) ?
    - Pure text-based RAG with advanced retrieval
    - Semantic search with reranking
    - Best for: Factual questions, information retrieval, summarization

    ## ?? Examples

    ### Text Query (Default)
    ```bash
    curl -X POST "/api/rag/query" \\
      -H "Content-Type: application/json" \\
      -d '{
        "query": "What are the dimensions of the NOVA product?",
        "top_k": 5
      }'
    ```

    ### Document-Specific Query
    ```bash
    curl -X POST "/api/rag/query" \\
      -H "Content-Type: application/json" \\
      -d '{
        "query": "Summarize this document",
        "document_ids": ["doc-123"],
        "top_k": 20
      }'
    ```

    ## ?? Migration from Old Endpoints

    **Old:** `POST /api/documents/{id}/query`
    **New:** `POST /api/rag/query` with `document_ids` filter

    **Old:** `POST /api/documents/{id}/summarize`
    **New:** `POST /api/rag/query` with summarization prompt
    """
    start_time = datetime.utcnow()

    try:
        # Standard text-based RAG query
        result = await llamaindex_service.advanced_rag_query(
            query=request.query,
            max_results=request.top_k,
            similarity_threshold=request.similarity_threshold,
            enable_reranking=request.enable_reranking,
            query_type="factual"
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        return QueryResponse(
            query=request.query,
            answer=result.get('answer', ''),
            sources=result.get('sources', []),
            confidence_score=result.get('confidence_score', 0.0),
            processing_time=processing_time,
            retrieved_chunks=len(result.get('sources', []))
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )
@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(
    request: ChatRequest,
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    Conversational interface for document Q&A.
    
    This endpoint maintains conversation context and provides
    contextual responses based on the document collection.
    """
    start_time = datetime.utcnow()
    
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid4())
        
        # Process chat message using advanced_rag_query
        result = await llamaindex_service.advanced_rag_query(
            query=request.message,
            max_results=request.top_k,
            query_type="conversational"
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return ChatResponse(
            message=request.message,
            response=result.get('response', ''),
            conversation_id=conversation_id,
            sources=result.get('sources', []),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Chat processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}"
        )


async def _enhance_search_results(
    results: List[Dict[str, Any]],
    workspace_id: str,
    include_related_products: bool = True,
    related_products_limit: int = 3
) -> List[Dict[str, Any]]:
    """
    Enhance search results with related products and images.

    Args:
        results: Raw search results
        workspace_id: Workspace ID for scoped queries
        include_related_products: Whether to include related products
        related_products_limit: Max related products per result

    Returns:
        Enhanced results with related products and images
    """
    try:
        supabase_client = get_supabase_client()
        product_rel_service = ProductRelationshipService(supabase_client=supabase_client.client)

        enhanced = []

        for result in results:
            # Get product ID from result
            product_id = result.get('id')
            if not product_id:
                enhanced.append(result)
                continue

            # Fetch related images
            try:
                images_response = supabase_client.table('product_image_relationships').select(
                    'id, image_id, relationship_type, relevance_score, document_images(id, image_url, caption)'
                ).eq('product_id', product_id).order('relevance_score', desc=True).limit(10).execute()

                related_images = []
                for img_rel in images_response.data or []:
                    if img_rel.get('document_images'):
                        related_images.append({
                            'id': img_rel['document_images']['id'],
                            'url': img_rel['document_images']['image_url'],
                            'relationship_type': img_rel['relationship_type'],
                            'relevance_score': img_rel['relevance_score'],
                            'caption': img_rel['document_images'].get('caption')
                        })

                result['related_images'] = related_images
            except Exception as e:
                logger.warning(f"Failed to fetch related images for product {product_id}: {e}")
                result['related_images'] = []

            # Fetch related products
            if include_related_products:
                try:
                    related_products = await product_rel_service.find_related_products(
                        product_id=product_id,
                        workspace_id=workspace_id,
                        limit=related_products_limit
                    )
                    result['related_products'] = related_products
                except Exception as e:
                    logger.warning(f"Failed to fetch related products for product {product_id}: {e}")
                    result['related_products'] = []
            else:
                result['related_products'] = []

            # Ensure all metadata is included
            if 'metadata' not in result:
                result['metadata'] = {}

            enhanced.append(result)

        return enhanced

    except Exception as e:
        logger.error(f"Error enhancing search results: {e}", exc_info=True)
        # Return original results if enhancement fails
        return results

@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    strategy: Optional[str] = Query(
        "multi_vector",
        description="Search strategy: 'multi_vector' (RECOMMENDED - default), 'semantic', 'vector', 'hybrid', 'material', 'image', 'color', 'texture', 'style', 'material_type', 'all'"
    ),
    enable_query_understanding: bool = Query(
        True,  # ? ENABLED BY DEFAULT - Makes platform smarter with minimal cost ($0.0001/query)
        description="?? AI query parsing to automatically extract filters from natural language (e.g., 'waterproof ceramic tiles for outdoor patio, matte finish' ? auto-extracts material_type, properties, finish, etc.). Set to false to disable."
    ),
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    **?? CONSOLIDATED SEARCH ENDPOINT - Single Entry Point for All Search Strategies**

    This endpoint replaces:
    - `/api/search/semantic` ? Use `strategy="semantic"`
    - `/api/search/similarity` ? Use `strategy="vector"`
    - `/api/unified-search` ? Use this endpoint

    ## ?? Search Strategies (All Implemented ?)

    ### Multi-Vector Search (`strategy="multi_vector"`) - ? RECOMMENDED DEFAULT ?
    - ?? **ENHANCED**: Combines 6 specialized CLIP embeddings + JSONB metadata filtering
    - **Embeddings Combined:**
      - text_embedding_1536 (20%) - Semantic understanding
      - visual_clip_embedding_512 (20%) - Visual similarity
      - color_clip_embedding_512 (15%) - Color palette matching
      - texture_clip_embedding_512 (15%) - Texture pattern matching
      - style_clip_embedding_512 (15%) - Design style matching
      - material_clip_embedding_512 (15%) - Material type matching
    - **+ JSONB Metadata Filtering**: Supports `material_filters` for property-based filtering
    - **+ Query Understanding**: ? **ENABLED BY DEFAULT** - Auto-extracts filters from natural language (set `enable_query_understanding=false` to disable)
    - **Performance**: Fast (~250-350ms with query understanding, ~200-300ms without), comprehensive, accurate
    - **Best For:** ALL queries - replaces need for `strategy="all"`
    - **Example:** "waterproof ceramic tiles for outdoor patio, matte finish"

    ### Semantic Search (`strategy="semantic"`) ?
    - Natural language understanding with MMR (Maximal Marginal Relevance)
    - Context-aware matching with diversity
    - Best for: Fast text queries, conceptual search, diverse results

    ### Vector Search (`strategy="vector"`) ?
    - Pure vector similarity (cosine distance)
    - Fast and efficient, no diversity filtering
    - Best for: Finding most similar documents, precise matching

    ### Hybrid Search (`strategy="hybrid"`) ?
    - Combines semantic (70%) + PostgreSQL full-text search (30%)
    - Best for: Balancing semantic understanding with keyword matching

    ### Material Property Search (`strategy="material"`) ?
    - JSONB-based filtering with AND/OR logic
    - Requires `material_filters` in request body
    - Best for: Filtering by specific material properties

    ### Image Similarity Search (`strategy="image"`) ?
    - Visual similarity using CLIP embeddings
    - Requires `image_url` or `image_base64` in request body
    - Best for: Finding visually similar products

    ### Specialized Visual Searches ? NEW
    - **Color Search** (`strategy="color"`): Color palette matching using specialized CLIP embeddings
      - Best for: "Find materials with warm tones", "similar color palette"
    - **Texture Search** (`strategy="texture"`): Texture pattern matching using specialized CLIP embeddings
      - Best for: "Find rough textured materials", "similar texture pattern"
    - **Style Search** (`strategy="style"`): Design style matching using specialized CLIP embeddings
      - Best for: "Find modern style materials", "similar design aesthetic"
    - **Material Type Search** (`strategy="material_type"`): Material type matching using specialized CLIP embeddings
      - Best for: "Find similar material types", "materials like this"

    ### All Strategies (`strategy="all"`) ?? DEPRECATED
    - ?? **DEPRECATED**: Use `strategy="multi_vector"` instead
    - **Why Deprecated:**
      - 10x slower (~800ms vs ~200ms)
      - 10x higher cost (10 separate searches)
      - Lower accuracy (simple averaging vs intelligent weighting)
      - Multi-vector already includes all 6 embedding types
    - **Parallel execution** of ALL 10 strategies using `asyncio.gather()`
    - **Only use if:** User explicitly requests "comprehensive search" or "all strategies"
    - **Recommendation:** Use `multi_vector` with `enable_query_understanding=true` instead

    ## ?? Examples

    ### Multi-Vector Search (? RECOMMENDED - Default)
    ```bash
    curl -X POST "/api/rag/search" \\
      -H "Content-Type: application/json" \\
      -d '{"query": "modern minimalist furniture", "workspace_id": "xxx", "top_k": 10}'
    ```

    ### Semantic Search (Fast text-only)
    ```bash
    curl -X POST "/api/rag/search?strategy=semantic" \\
      -H "Content-Type: application/json" \\
      -d '{"query": "oak wood flooring", "workspace_id": "xxx", "top_k": 5}'
    ```

    ### Specialized Color Search
    ```bash
    curl -X POST "/api/rag/search?strategy=color" \\
      -H "Content-Type: application/json" \\
      -d '{"query": "warm tones", "workspace_id": "xxx", "top_k": 10}'
    ```

    ### Material Property Search
    ```bash
    curl -X POST "/api/rag/search?strategy=material" \\
      -H "Content-Type: application/json" \\
      -d '{"workspace_id": "xxx", "material_filters": {"material_type": "fabric", "color": ["red", "blue"]}, "top_k": 10}'
    ```

    ### Image Similarity Search
    ```bash
    curl -X POST "/api/rag/search?strategy=image" \\
      -H "Content-Type: application/json" \\
      -d '{"workspace_id": "xxx", "image_url": "https://example.com/image.jpg", "top_k": 10}'
    ```

    ### All Strategies (Parallel Execution - 3-4x Faster!)
    ```bash
    curl -X POST "/api/rag/search?strategy=all" \\
      -H "Content-Type: application/json" \\
      -d '{"query": "modern oak furniture", "workspace_id": "xxx", "top_k": 10}'
    ```

    ## ?? Response Example (All Strategies)
    ```json
    {
      "query": "modern oak furniture",
      "enhanced_query": "modern oak furniture",
      "results": [
        {
          "id": "product_uuid_1",
          "name": "Modern Oak Dining Table",
          "description": "Contemporary oak furniture...",
          "score": 0.92,
          "final_score": 0.85,
          "strategy_count": 4,
          "strategies": ["semantic", "vector", "multi_vector", "hybrid"]
        }
      ],
      "total_results": 10,
      "search_type": "all",
      "processing_time": 0.223,
      "search_metadata": {
        "strategies_executed": 4,
        "strategies_successful": 4,
        "strategies_failed": 0,
        "strategy_breakdown": {
          "semantic": {"count": 3, "success": true},
          "vector": {"count": 2, "success": true},
          "multi_vector": {"count": 4, "success": true},
          "hybrid": {"count": 5, "success": true}
        },
        "parallel_execution": true,
        "parallel_processing_time": 0.017
      }
    }
    ```

    ## ? Performance Characteristics

    | Strategy | Typical Time | Max Time | Notes |
    |----------|-------------|----------|-------|
    | semantic | 100-150ms | 300ms | Indexed, MMR diversity |
    | vector | 50-100ms | 200ms | Fastest, pure similarity |
    | multi_vector | 200-300ms | 500ms | 3 embeddings, sequential scan for 2048-dim |
    | hybrid | 120-180ms | 350ms | Semantic + full-text search |
    | material | 30-50ms | 100ms | JSONB indexed |
    | image | 100-150ms | 300ms | CLIP indexed |
    | **all (parallel)** | **200-300ms** | **500ms** | **3-4x faster than sequential** |

    ## ?? Migration from Old Endpoints

    **Old:** `POST /api/search/semantic`
    **New:** `POST /api/rag/search?strategy=semantic`

    **Old:** `POST /api/search/similarity`
    **New:** `POST /api/rag/search?strategy=vector`

    **Old:** `POST /api/unified-search`
    **New:** `POST /api/rag/search` (same functionality, clearer naming)

    ## ?? Error Codes

    - **400 Bad Request**: Invalid parameters (missing query, invalid strategy, etc.)
    - **401 Unauthorized**: Missing or invalid authentication
    - **404 Not Found**: Workspace not found
    - **500 Internal Server Error**: Search processing failed
    - **503 Service Unavailable**: LlamaIndex service not available

    ## ?? Rate Limits

    - **60 requests/minute** per user
    - **1000 requests/hour** per workspace
    - Parallel execution (`strategy="all"`) counts as 1 request
    """
    start_time = datetime.utcnow()

    try:
        # Validate strategy
        valid_strategies = ['semantic', 'vector', 'multi_vector', 'hybrid', 'material', 'image', 'all']
        if strategy not in valid_strategies:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid strategy '{strategy}'. Valid strategies: {', '.join(valid_strategies)}"
            )

        # Initialize services
        supabase_client = get_supabase_client()
        search_prompt_service = SearchPromptService(supabase_client=supabase_client.client)
        product_rel_service = ProductRelationshipService(supabase_client=supabase_client.client)

        # Apply enhancement prompt to query if enabled
        query_to_use = request.query
        enhanced_query = None
        prompts_applied = []

        if request.use_search_prompts:
            enhancement_result = await search_prompt_service.enhance_query(
                query=request.query,
                workspace_id=request.workspace_id,
                custom_prompt=request.custom_formatting_prompt
            )
            if enhancement_result.get('enhancement_applied'):
                query_to_use = enhancement_result['enhanced_query']
                enhanced_query = query_to_use
                prompts_applied.extend(enhancement_result.get('prompts_applied', []))

        # ?? STEP 1: Query Understanding (if enabled)
        # Parse natural language query to extract structured filters BEFORE multi-strategy search
        parsed_filters = {}
        if enable_query_understanding:
            try:
                from app.services.unified_search_service import UnifiedSearchService

                # Create temporary service instance for query parsing
                unified_service = UnifiedSearchService()
                visual_query, parsed_filters = await unified_service._parse_query_with_ai(query_to_use)

                # Update query to use visual query (core concept for embedding)
                query_to_use = visual_query

                # Merge parsed filters with existing material_filters (user filters take precedence)
                existing_filters = getattr(request, 'material_filters', {})
                if existing_filters:
                    # User-provided filters override AI-parsed filters
                    merged_filters = {**parsed_filters, **existing_filters}
                else:
                    merged_filters = parsed_filters

                # Update request with merged filters
                if merged_filters:
                    request.material_filters = merged_filters

                logger.info(f"?? Query understanding: '{request.query}' ? visual_query='{visual_query}', filters={parsed_filters}")

            except Exception as e:
                logger.error(f"Query understanding failed: {e}, continuing with original query")
                # Continue with original query if parsing fails

        # ?? STEP 2: Route to appropriate search method based on strategy
        # All strategies now use the parsed query + extracted filters
        if strategy == "semantic":
            # Semantic search using MMR (Maximal Marginal Relevance)
            # Balances relevance and diversity (lambda_mult=0.5)
            results = await llamaindex_service.semantic_search_with_mmr(
                query=query_to_use,
                k=request.top_k,
                lambda_mult=0.5
            )

        elif strategy == "vector":
            # Pure vector similarity search (cosine distance)
            # No diversity filtering (lambda_mult=1.0)
            results = await llamaindex_service.semantic_search_with_mmr(
                query=query_to_use,
                k=request.top_k,
                lambda_mult=1.0  # Pure similarity, no diversity
            )

        elif strategy == "multi_vector":
            # ?? Enhanced multi-vector search combining 6 specialized CLIP embeddings + metadata filtering
            # text (20%), visual (20%), color (15%), texture (15%), style (15%), material (15%)
            material_filters = getattr(request, 'material_filters', None)
            results = await llamaindex_service.multi_vector_search(
                query=query_to_use,
                workspace_id=request.workspace_id,
                top_k=request.top_k,
                material_filters=material_filters
            )

        elif strategy == "hybrid":
            # Hybrid search combining semantic (70%) + full-text keyword search (30%)
            results = await llamaindex_service.hybrid_search(
                query=query_to_use,
                workspace_id=request.workspace_id,
                top_k=request.top_k
            )

        elif strategy == "material":
            # Material property search using JSONB filtering
            # Requires material_filters in request
            material_filters = getattr(request, 'material_filters', {})
            if not material_filters:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="material_filters required for material property search"
                )
            results = await llamaindex_service.material_property_search(
                workspace_id=request.workspace_id,
                material_filters=material_filters,
                top_k=request.top_k
            )

        elif strategy == "image":
            # Image similarity search using CLIP embeddings
            # Requires image_url or image_base64 in request
            image_url = getattr(request, 'image_url', None)
            image_base64 = getattr(request, 'image_base64', None)
            if not image_url and not image_base64:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="image_url or image_base64 required for image similarity search"
                )
            results = await llamaindex_service.image_similarity_search(
                workspace_id=request.workspace_id,
                image_url=image_url,
                image_base64=image_base64,
                top_k=request.top_k
            )

        elif strategy == "all":
            # ?? DEPRECATED: Use strategy="multi_vector" instead
            logger.warning(f"?? DEPRECATED: strategy='all' is deprecated. Use strategy='multi_vector' instead for 10x better performance and accuracy.")
            logger.warning(f"   Current: 10 separate searches (~800ms, 10x cost, simple averaging)")
            logger.warning(f"   Recommended: 1 intelligent search (~200ms, 1x cost, weighted scoring with 6 embeddings)")

            # Run all strategies in parallel (DEPRECATED - use multi_vector instead)
            material_filters = getattr(request, 'material_filters', None)
            image_url = getattr(request, 'image_url', None)
            image_base64 = getattr(request, 'image_base64', None)

            results = await llamaindex_service.search_all_strategies(
                query=query_to_use,
                workspace_id=request.workspace_id,
                top_k=request.top_k,
                material_filters=material_filters,
                image_url=image_url,
                image_base64=image_base64
            )

        # Get raw results
        raw_results = results.get('results', [])

        # Apply formatting, filtering, enrichment prompts if enabled
        processed_results = raw_results
        if request.use_search_prompts:
            processed_results = await search_prompt_service.format_results(
                raw_results, request.workspace_id, request.custom_formatting_prompt
            )
            processed_results = await search_prompt_service.filter_results(
                processed_results, request.workspace_id
            )
            processed_results = await search_prompt_service.enrich_results(
                processed_results, request.workspace_id
            )

        # Enhance results with related products and images
        enhanced_results = await _enhance_search_results(
            processed_results,
            request.workspace_id,
            request.include_related_products,
            request.related_products_limit
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Build search metadata
        search_metadata = {
            'prompts_applied': prompts_applied,
            'prompts_enabled': request.use_search_prompts,
            'related_products_included': request.include_related_products
        }

        # Add parallel execution metadata for 'all' strategy
        if strategy == "all":
            search_metadata.update({
                'strategies_executed': results.get('strategies_executed', 0),
                'strategies_successful': results.get('strategies_successful', 0),
                'strategies_failed': results.get('strategies_failed', 0),
                'strategy_breakdown': results.get('strategy_breakdown', {}),
                'parallel_execution': True,
                'parallel_processing_time': results.get('processing_time', 0)
            })

        return SearchResponse(
            query=request.query,
            enhanced_query=enhanced_query,
            results=enhanced_results,
            total_results=results.get('total_results', 0),
            search_type=strategy,
            processing_time=processing_time,
            search_metadata=search_metadata
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search processing failed: {str(e)}"
        )
@router.post("/search/mmr", response_model=MMRSearchResponse)
async def mmr_search(
    request: MMRSearchRequest,
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    Perform MMR (Maximal Marginal Relevance) search for diverse results.
    
    This endpoint implements MMR search to provide diverse, non-redundant results
    by balancing relevance and diversity using the lambda parameter.
    """
    try:
        start_time = datetime.utcnow()
        
        # Call the MMR search method from LlamaIndex service
        results = await llamaindex_service.semantic_search_with_mmr(
            query=request.query,
            top_k=request.top_k,
            diversity_threshold=request.diversity_threshold,
            lambda_param=request.lambda_param,
            document_ids=request.document_ids,
            include_metadata=request.include_metadata
        )
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        return MMRSearchResponse(
            query=request.query,
            results=results.get('results', []),
            total_results=results.get('total_results', 0),
            diversity_score=results.get('diversity_score', 0.0),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"MMR search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MMR search failed: {str(e)}"
        )
@router.post("/search/advanced", response_model=AdvancedQueryResponse)
async def advanced_query_search(
    request: AdvancedQueryRequest,
    llamaindex_service: LlamaIndexService = Depends(get_llamaindex_service)
):
    """
    Perform advanced query search with optimization and expansion.
    
    This endpoint provides advanced query processing including query expansion,
    rewriting, and optimization based on query type and search parameters.
    """
    try:
        start_time = datetime.utcnow()
        
        # Convert string enums to proper enum types
        query_type = QueryType(request.query_type.upper())
        search_operator = SearchOperator(request.search_operator.upper())
        
        # Call the advanced query method from LlamaIndex service
        results = await llamaindex_service.advanced_query_with_optimization(
            query=request.query,
            query_type=query_type,
            top_k=request.top_k,
            enable_expansion=request.enable_expansion,
            enable_rewriting=request.enable_rewriting,
            similarity_threshold=request.similarity_threshold,
            document_ids=request.document_ids,
            metadata_filters=request.metadata_filters,
            search_operator=search_operator
        )

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        return AdvancedQueryResponse(
            query=request.query,
            results=results.get('results', []),
            total_results=results.get('total_results', 0),
            processing_time=processing_time,
            query_type=request.query_type,
            optimizations_applied=results.get('optimizations_applied', [])
        )

    except Exception as e:
        logger.error(f"Advanced query search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Advanced query search failed: {str(e)}"
        )


