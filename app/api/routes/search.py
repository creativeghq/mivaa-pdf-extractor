"""
Search Routes - Query, chat, and search endpoints
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import uuid4
from fastapi import APIRouter, HTTPException, Depends, status, Query

from app.services.rag_service import RAGService, get_rag_service
from app.services.advanced_search_service import AdvancedSearchService, QueryType, SearchOperator
from app.services.product_relationship_service import ProductRelationshipService
from app.services.search_prompt_service import SearchPromptService
from .models import (
    QueryRequest, QueryResponse,
    ChatRequest, ChatResponse,
    SearchRequest, SearchResponse,
    MMRSearchRequest, MMRSearchResponse,
    AdvancedQueryRequest, AdvancedQueryResponse
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/query", response_model=QueryResponse, deprecated=True)
async def query_documents(
    request: QueryRequest
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
    raise HTTPException(
        status_code=410,
        detail="Endpoint deprecated. Use /api/rag/search with multi_vector strategy."
    )

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
@router.post("/chat", response_model=ChatResponse, deprecated=True)
async def chat_with_documents(
    request: ChatRequest
):
    """
    Conversational interface for document Q&A.
    
    This endpoint maintains conversation context and provides
    contextual responses based on the document collection.
    """
    raise HTTPException(
        status_code=410,
        detail="Endpoint deprecated. Use /api/rag/search with multi_vector strategy."
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
    rag_service: RAGService = Depends(get_rag_service),
    strategy: Optional[str] = Query(
        "multi_vector",
        description="Search strategy: only 'multi_vector' is supported"
    ),
    enable_query_understanding: bool = Query(
        True,
        description="AI query parsing to automatically extract filters from natural language. Enabled by default."
    )
):
    """
    **Multi-Vector Search Endpoint - TRUE 6-Embedding Fusion**

    Combines ALL 6 specialized embeddings in parallel for maximum accuracy:

    **Embedding Weights:**
    - text (20%) - Semantic text understanding from metadata/descriptions
    - visual (20%) - General visual similarity (SigLIP 1152D)
    - color (15%) - Color palette matching (specialized CLIP)
    - texture (15%) - Texture pattern matching (specialized CLIP)
    - style (15%) - Design style matching (specialized CLIP)
    - material (15%) - Material type matching (specialized CLIP)

    **How it works:**
    1. Generate visual embedding from query text
    2. Search all 5 VECS collections in parallel (visual, color, texture, style, material)
    3. Map image results to products via product_image_relationships
    4. Combine scores with intelligent weighting
    5. Apply metadata filters as soft boosts

    **Features:**
    - AI query understanding (enabled by default) - Auto-extracts filters from natural language
    - JSONB metadata filtering via `material_filters`
    - Metadata prototype validation scoring
    - Performance: ~300-500ms (parallel execution)

    **Example:**
    ```bash
    curl -X POST "/api/rag/search" \
      -H "Content-Type: application/json" \
      -d '{"query": "waterproof ceramic tiles for outdoor patio", "workspace_id": "xxx", "top_k": 10}'
    ```

    **Response includes individual embedding scores:**
    ```json
    {
      "results": [{
        "id": "product_123",
        "score": 0.87,
        "text_score": 0.85,
        "visual_score": 0.92,
        "color_score": 0.88,
        "texture_score": 0.81,
        "style_score": 0.79,
        "material_score": 0.90
      }],
      "total_results": 10,
      "processing_time": 0.345,
      "embeddings_used": ["text", "visual", "color", "texture", "style", "material"]
    }
    ```
    """
    start_time = datetime.utcnow()

    try:
        # Only multi_vector strategy is supported
        if strategy != 'multi_vector':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid strategy '{strategy}'. Only 'multi_vector' is supported."
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

        # Execute multi-vector search (the only supported strategy)
        # Combines 6 specialized CLIP embeddings + metadata filtering
        # text (20%), visual (20%), color (15%), texture (15%), style (15%), material (15%)
        material_filters = getattr(request, 'material_filters', None)
        results = await rag_service.multi_vector_search(
            query=query_to_use,
            workspace_id=request.workspace_id,
            top_k=request.top_k,
            material_filters=material_filters
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
            'related_products_included': request.include_related_products,
            'strategy': 'multi_vector'
        }

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
@router.post("/search/mmr", response_model=MMRSearchResponse, deprecated=True)
async def mmr_search(
    request: MMRSearchRequest
):
    """
    Perform MMR (Maximal Marginal Relevance) search for diverse results.
    
    This endpoint implements MMR search to provide diverse, non-redundant results
    by balancing relevance and diversity using the lambda parameter.
    """
    try:
        start_time = datetime.utcnow()
        
        raise HTTPException(status_code=410, detail="Endpoint deprecated. Use /api/rag/search.")
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
@router.post("/search/advanced", response_model=AdvancedQueryResponse, deprecated=True)
async def advanced_query_search(
    request: AdvancedQueryRequest
):
    """
    Perform advanced query search with optimization and expansion.
    
    This endpoint provides advanced query processing including query expansion,
    rewriting, and optimization based on query type and search parameters.
    """
    try:
        start_time = datetime.utcnow()
        
        raise HTTPException(status_code=410, detail="Endpoint deprecated. Use /api/rag/search.")

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


