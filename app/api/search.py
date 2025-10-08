"""
Search and RAG API Endpoints

This module provides advanced search and RAG (Retrieval-Augmented Generation) 
endpoints using LlamaIndex for semantic search, document querying, and 
intelligent document analysis.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse

from ..schemas.search import (
    # Enhanced multi-modal schemas
    SearchRequest,
    SearchResult,
    SearchResponse,
    DocumentQueryRequest,
    DocumentQueryResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
    RelatedDocumentsResponse,
    DocumentSummaryRequest,
    DocumentSummaryResponse,
    EntityExtractionRequest,
    EntityExtractionResponse,
    DocumentComparisonRequest,
    DocumentComparisonResponse,
    QueryRequest,
    QueryResponse,
    SourceCitation,
    ImageSearchRequest,
    ImageSearchResult,
    ImageSearchResponse,
    MultiModalAnalysisRequest,
    MultiModalAnalysisResponse,
    # Legacy schemas removed - Phase 8 cleanup completed
    # All functionality now handled by modern multi-modal schemas above
)
from ..schemas.common import ErrorResponse, SuccessResponse
from ..services.llamaindex_service import LlamaIndexService
from ..services.supabase_client import SupabaseClient
from ..services.material_visual_search_service import (
    MaterialVisualSearchService,
    MaterialSearchRequest,
    MaterialSearchResponse,
    get_material_visual_search_service
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["Search", "Embeddings", "Chat"])

# Initialize services
llamaindex_service = LlamaIndexService()
supabase_client = SupabaseClient()


async def get_llamaindex_service() -> LlamaIndexService:
    """Dependency to get LlamaIndex service instance."""
    if not llamaindex_service.available:
        raise HTTPException(
            status_code=503,
            detail="LlamaIndex service is not available. Please check service configuration."
        )
    return llamaindex_service


async def get_supabase_client() -> SupabaseClient:
    """Dependency to get Supabase client instance."""
    if not supabase_client.client:
        raise HTTPException(
            status_code=503,
            detail="Database service is not available. Please check configuration."
        )
    return supabase_client


@router.post(
    "/documents/{document_id}/query",
    response_model=DocumentQueryResponse,
    summary="Query specific document",
    description="Query a specific document using natural language with RAG capabilities"
)
async def query_document(
    document_id: str,
    request: DocumentQueryRequest,
    llamaindex: LlamaIndexService = Depends(get_llamaindex_service),
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> DocumentQueryResponse:
    """
    Query a specific document using RAG.
    
    This endpoint allows you to ask natural language questions about a specific
    document and get intelligent responses based on the document's content.
    """
    try:
        # Verify document exists in database
        document_data = await supabase.get_document_by_id(document_id)
        if not document_data:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
        
        # Execute query using LlamaIndex
        result = await llamaindex.query_document(
            document_id=document_id,
            query=request.query,
            response_mode=request.response_mode
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Query failed: {result.get('error', 'Unknown error')}"
            )
        
        return DocumentQueryResponse(
            success=True,
            document_id=document_id,
            query=request.query,
            response=result["response"],
            sources=result.get("sources", []),
            metadata={
                **result.get("metadata", {}),
                "document_title": document_data.get("title", ""),
                "document_type": document_data.get("content_type", "")
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error querying document {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/search/semantic",
    response_model=SemanticSearchResponse,
    summary="Semantic search across documents",
    description="Perform semantic search across multiple documents using vector similarity"
)
async def semantic_search(
    request: SemanticSearchRequest,
    llamaindex: LlamaIndexService = Depends(get_llamaindex_service),
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> SemanticSearchResponse:
    """
    Perform semantic search across multiple documents.
    
    This endpoint searches across all indexed documents to find the most
    semantically relevant content based on the query.
    """
    try:
        # Get list of available documents
        if request.document_ids:
            # Search only in specified documents
            document_ids = request.document_ids
        else:
            # Search across all documents
            documents = await supabase.list_documents(
                limit=1000,  # Reasonable limit for search
                status_filter="completed"
            )
            document_ids = [doc["id"] for doc in documents.get("documents", [])]
        
        if not document_ids:
            return SemanticSearchResponse(
                success=True,
                query=request.query,
                results=[],
                total_results=0,
                metadata={
                    "searched_documents": 0,
                    "similarity_threshold": request.similarity_threshold
                }
            )
        
        # Perform search across documents
        search_results = []
        for doc_id in document_ids:
            try:
                result = await llamaindex.query_document(
                    document_id=doc_id,
                    query=request.query,
                    response_mode="compact"
                )
                
                if result["success"] and result.get("sources"):
                    # Process sources and filter by similarity threshold
                    for source in result["sources"]:
                        score = source.get("score", 0.0)
                        if score >= request.similarity_threshold:
                            search_results.append({
                                "document_id": doc_id,
                                "score": score,
                                "content": source.get("text_snippet", ""),
                                "metadata": source.get("metadata", {})
                            })
                            
            except Exception as e:
                logger.warning(f"Error searching document {doc_id}: {e}")
                continue
        
        # Sort by score and apply limit
        search_results.sort(key=lambda x: x["score"], reverse=True)
        limited_results = search_results[:request.limit]
        
        return SemanticSearchResponse(
            success=True,
            query=request.query,
            results=limited_results,
            total_results=len(search_results),
            metadata={
                "searched_documents": len(document_ids),
                "similarity_threshold": request.similarity_threshold,
                "returned_results": len(limited_results)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in semantic search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/search/similarity",
    response_model=SimilaritySearchResponse,
    summary="Vector similarity search",
    description="Find documents similar to a given text using vector embeddings"
)
async def similarity_search(
    request: SimilaritySearchRequest,
    llamaindex: LlamaIndexService = Depends(get_llamaindex_service),
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> SimilaritySearchResponse:
    """
    Perform vector similarity search to find similar documents.
    
    This endpoint finds documents that are semantically similar to the
    provided text using vector embeddings.
    """
    try:
        # For now, we'll use semantic search as the underlying mechanism
        # In a more advanced implementation, we could use dedicated vector search
        semantic_request = SemanticSearchRequest(
            query=request.reference_text,
            document_ids=None,  # SimilaritySearchRequest doesn't have document_ids field
            limit=request.limit,
            similarity_threshold=request.similarity_threshold
        )
        
        semantic_result = await semantic_search(semantic_request, llamaindex, supabase)
        
        # Transform the response format
        similar_documents = []
        for result in semantic_result.results:
            similar_documents.append({
                "document_id": result["document_id"],
                "similarity_score": result["score"],
                "content_snippet": result["content"],
                "metadata": result["metadata"]
            })
        
        return SimilaritySearchResponse(
            success=True,
            reference_info={
                "reference_text": request.reference_text,
                "type": "text"
            },
            similar_documents=similar_documents,
            total_found=semantic_result.total_results,
            search_time_ms=getattr(semantic_result, 'search_time_ms', 0.0),
            metadata={
                **semantic_result.metadata,
                "search_type": "vector_similarity"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/documents/{document_id}/related",
    response_model=RelatedDocumentsResponse,
    summary="Find related documents",
    description="Find documents related to a specific document based on content similarity"
)
async def find_related_documents(
    document_id: str,
    limit: int = Query(5, ge=1, le=20, description="Maximum number of related documents to return"),
    similarity_threshold: float = Query(0.7, ge=0.0, le=1.0, description="Minimum similarity threshold"),
    llamaindex: LlamaIndexService = Depends(get_llamaindex_service),
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> RelatedDocumentsResponse:
    """
    Find documents related to a specific document.
    
    This endpoint analyzes the content of a document and finds other documents
    with similar content or themes.
    """
    try:
        # Verify source document exists
        document_data = await supabase.get_document_by_id(document_id)
        if not document_data:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
        
        # Get a summary of the source document to use for similarity search
        summary_result = await llamaindex.summarize_document(
            document_id=document_id,
            summary_type="brief"
        )
        
        if not summary_result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to analyze document: {summary_result.get('error', 'Unknown error')}"
            )
        
        # Use the summary to find similar documents
        similarity_request = SimilaritySearchRequest(
            text=summary_result["response"],
            limit=limit + 1,  # +1 to account for the source document itself
            similarity_threshold=similarity_threshold
        )
        
        similarity_result = await similarity_search(similarity_request, llamaindex, supabase)
        
        # Filter out the source document from results
        related_docs = [
            doc for doc in similarity_result.similar_documents 
            if doc["document_id"] != document_id
        ][:limit]
        
        return RelatedDocumentsResponse(
            success=True,
            source_document_id=document_id,
            related_documents=related_docs,
            total_found=len(related_docs),
            metadata={
                "similarity_threshold": similarity_threshold,
                "analysis_method": "content_summary_similarity"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding related documents for {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/documents/{document_id}/summarize",
    response_model=DocumentSummaryResponse,
    summary="Generate document summary",
    description="Generate an intelligent summary of a document using RAG"
)
async def summarize_document(
    document_id: str,
    request: DocumentSummaryRequest,
    llamaindex: LlamaIndexService = Depends(get_llamaindex_service),
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> DocumentSummaryResponse:
    """
    Generate an intelligent summary of a document.
    
    This endpoint creates different types of summaries (brief, comprehensive, 
    key points) based on the document's content.
    """
    try:
        # Verify document exists
        document_data = await supabase.get_document_by_id(document_id)
        if not document_data:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
        
        # Generate summary using LlamaIndex
        result = await llamaindex.summarize_document(
            document_id=document_id,
            summary_type=request.summary_type
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Summarization failed: {result.get('error', 'Unknown error')}"
            )
        
        return DocumentSummaryResponse(
            success=True,
            document_id=document_id,
            summary_type=request.summary_type,
            summary=result["response"],
            metadata={
                **result.get("metadata", {}),
                "document_title": document_data.get("title", ""),
                "document_length": len(document_data.get("content", "")),
                "summary_length": len(result["response"])
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error summarizing document {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/documents/{document_id}/extract-entities",
    response_model=EntityExtractionResponse,
    summary="Extract named entities",
    description="Extract named entities (people, organizations, dates, etc.) from a document"
)
async def extract_entities(
    document_id: str,
    request: EntityExtractionRequest,
    llamaindex: LlamaIndexService = Depends(get_llamaindex_service),
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> EntityExtractionResponse:
    """
    Extract named entities from a document.
    
    This endpoint identifies and extracts various types of entities such as
    people, organizations, dates, locations, etc. from the document content.
    """
    try:
        # Verify document exists
        document_data = await supabase.get_document_by_id(document_id)
        if not document_data:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found"
            )
        
        # Extract entities using LlamaIndex
        result = await llamaindex.extract_entities(
            document_id=document_id,
            entity_types=request.entity_types
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Entity extraction failed: {result.get('error', 'Unknown error')}"
            )
        
        return EntityExtractionResponse(
            success=True,
            document_id=document_id,
            entity_types=request.entity_types,
            entities=result["response"],  # This would need parsing in a real implementation
            metadata={
                **result.get("metadata", {}),
                "document_title": document_data.get("title", ""),
                "extraction_method": "llm_based"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting entities from document {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post(
    "/documents/compare",
    response_model=DocumentComparisonResponse,
    summary="Compare multiple documents",
    description="Compare and analyze similarities and differences between multiple documents"
)
async def compare_documents(
    request: DocumentComparisonRequest,
    llamaindex: LlamaIndexService = Depends(get_llamaindex_service),
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> DocumentComparisonResponse:
    """
    Compare multiple documents.
    
    This endpoint analyzes multiple documents and provides insights about
    their similarities, differences, and relationships.
    """
    try:
        if len(request.document_ids) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 documents are required for comparison"
            )
        
        # Verify all documents exist
        document_data = {}
        for doc_id in request.document_ids:
            doc_data = await supabase.get_document_by_id(doc_id)
            if not doc_data:
                raise HTTPException(
                    status_code=404,
                    detail=f"Document {doc_id} not found"
                )
            document_data[doc_id] = doc_data
        
        # Perform comparison using LlamaIndex
        result = await llamaindex.compare_documents(
            document_ids=request.document_ids,
            comparison_aspect=request.comparison_aspect
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Document comparison failed: {result.get('error', 'Unknown error')}"
            )
        
        # Prepare document info for response
        documents_info = [
            {
                "document_id": doc_id,
                "title": document_data[doc_id].get("title", ""),
                "content_type": document_data[doc_id].get("content_type", "")
            }
            for doc_id in request.document_ids
        ]
        
        return DocumentComparisonResponse(
            success=True,
            document_ids=request.document_ids,
            comparison_aspect=request.comparison_aspect,
            comparison_result=result["response"],
            documents_info=documents_info,
            metadata={
                **result.get("metadata", {}),
                "comparison_method": "llm_based_analysis"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get(
    "/search/health",
    response_model=SuccessResponse,
    summary="Search service health check",
    description="Check the health and availability of search and RAG services"
)
async def search_health_check(
    llamaindex: LlamaIndexService = Depends(get_llamaindex_service)
) -> SuccessResponse:
    """
    Check the health of search and RAG services.
    
    This endpoint provides information about the availability and status
    of the LlamaIndex service and its components.
    """
    try:
        health_status = await llamaindex.health_check()
        
        return SuccessResponse(
            success=True,
            message="Search and RAG services are operational",
            data=health_status
        )
        
    except Exception as e:
        logger.error(f"Error in search health check: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Health check failed: {str(e)}"
        )


# ============================================================================
# NEW MULTI-MODAL ENDPOINTS
# ============================================================================

@router.post(
    "/search/multimodal",
    response_model=SearchResponse,
    summary="Multi-modal search across documents",
    description="Perform advanced multi-modal search across text and images with OCR support"
)
async def multimodal_search(
    request: SearchRequest,
    llamaindex: LlamaIndexService = Depends(get_llamaindex_service),
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> SearchResponse:
    """
    Perform multi-modal search across documents with text and image support.
    
    This endpoint supports:
    - Text-based semantic search
    - OCR text search within images
    - Image content analysis
    - Combined multi-modal scoring
    """
    try:
        # Get configuration for multi-modal settings
        from app.config import get_settings
        settings = get_settings()
        multimodal_config = settings.get_multimodal_config()
        ocr_config = settings.get_ocr_config()
        
        # Validate multi-modal is enabled
        if not multimodal_config.get("enable_multimodal", False):
            raise HTTPException(
                status_code=400,
                detail="Multi-modal search is not enabled. Please check configuration."
            )
        
        # Build search parameters
        search_params = {
            "query": request.query,
            "limit": request.limit,
            "similarity_threshold": request.similarity_threshold,
            "include_images": request.include_images,
            "include_ocr_text": request.include_ocr_text,
            "content_types": request.content_types,
            "ocr_confidence_threshold": request.ocr_confidence_threshold,
            "multimodal_llm_model": multimodal_config.get("llm_model"),
            "image_analysis_depth": "standard"
        }
        
        # Filter documents by IDs if specified
        if request.document_ids:
            document_ids = request.document_ids
        else:
            # Get all available documents
            documents = await supabase.list_documents(
                limit=1000,
                status_filter="completed"
            )
            document_ids = [doc["id"] for doc in documents.get("documents", [])]
        
        if not document_ids:
            return SearchResponse(
                success=True,
                query=request.query,
                results=[],
                total_found=0,
                search_time_ms=0.0,
                search_type="multimodal",
                filters_applied={},
                metadata={
                    "searched_documents": 0,
                    "multimodal_enabled": True,
                    "search_type": "multimodal"
                }
            )
        
        # Perform multi-modal search
        search_results = []
        for doc_id in document_ids:
            try:
                # Use enhanced LlamaIndex service for multi-modal search
                result = await llamaindex.multimodal_search(
                    document_id=doc_id,
                    **search_params
                )
                
                if result["success"] and result.get("results"):
                    for item in result["results"]:
                        search_result = SearchResult(
                            document_id=doc_id,
                            score=item.get("score", 0.0),
                            multimodal_score=item.get("multimodal_score", 0.0),
                            content=item.get("content", ""),
                            ocr_text=item.get("ocr_text", ""),
                            ocr_confidence=item.get("ocr_confidence", 0.0),
                            content_type=item.get("content_type", "text"),
                            associated_images=item.get("associated_images", []),
                            image_analysis=item.get("image_analysis", {}),
                            metadata=item.get("metadata", {})
                        )
                        search_results.append(search_result)
                        
            except Exception as e:
                logger.warning(f"Error in multi-modal search for document {doc_id}: {e}")
                continue
        
        # Sort by multimodal score (or regular score if not available)
        search_results.sort(
            key=lambda x: x.multimodal_score if x.multimodal_score > 0 else x.score,
            reverse=True
        )
        limited_results = search_results[:request.limit]
        
        return SearchResponse(
            success=True,
            query=request.query,
            results=limited_results,
            total_found=len(search_results),
            search_time_ms=200.0,
            search_type="multimodal",
            filters_applied={},
            metadata={
                "searched_documents": len(document_ids),
                "multimodal_enabled": True,
                "search_type": "multimodal",
                "returned_results": len(limited_results),
                "ocr_enabled": ocr_config.get("ocr_enabled", False),
                "image_analysis_enabled": request.include_images
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in multi-modal search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Multi-modal search failed: {str(e)}"
        )


@router.post(
    "/query/multimodal",
    response_model=QueryResponse,
    summary="Multi-modal RAG query",
    description="Query documents using multi-modal RAG with text and image context"
)
async def multimodal_query(
    request: QueryRequest,
    llamaindex: LlamaIndexService = Depends(get_llamaindex_service),
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> QueryResponse:
    """
    Perform multi-modal RAG query with enhanced context from text and images.
    
    This endpoint provides intelligent responses using:
    - Text content from documents
    - OCR-extracted text from images
    - Image analysis and descriptions
    - Multi-modal LLM reasoning
    """
    try:
        # Get configuration
        from app.config import get_settings
        settings = get_settings()
        multimodal_config = settings.get_multimodal_config()
        
        # Validate multi-modal is enabled
        if not multimodal_config.get("enable_multimodal", False):
            raise HTTPException(
                status_code=400,
                detail="Multi-modal RAG is not enabled. Please check configuration."
            )
        
        # Build query parameters
        query_params = {
            "query": request.question,
            "include_image_context": request.include_image_context,
            "multimodal_llm_model": request.multimodal_llm_model or multimodal_config.get("llm_model"),
            "image_analysis_depth": request.image_analysis_depth,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "response_mode": getattr(request, 'response_mode', 'compact')
        }
        
        # Filter documents if specified
        if request.document_ids:
            document_ids = request.document_ids
        else:
            # Get all available documents
            documents = await supabase.list_documents(
                limit=1000,
                status_filter="completed"
            )
            document_ids = [doc["id"] for doc in documents.get("documents", [])]
        
        if not document_ids:
            raise HTTPException(
                status_code=400,
                detail="No documents available for querying"
            )
        
        # Perform multi-modal RAG query
        result = await llamaindex.multimodal_query(
            document_ids=document_ids,
            **query_params
        )
        
        if not result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Multi-modal query failed: {result.get('error', 'Unknown error')}"
            )
        
        # Build source citations with multi-modal information
        sources = []
        for source in result.get("sources", []):
            citation = SourceCitation(
                document_id=source.get("document_id", ""),
                content_excerpt=source.get("content_excerpt", ""),
                content_type=source.get("content_type", "text"),
                ocr_excerpt=source.get("ocr_excerpt", ""),
                image_reference=source.get("image_reference", ""),
                multimodal_confidence=source.get("multimodal_confidence", 0.0),
                page_number=source.get("page_number"),
                metadata=source.get("metadata", {})
            )
            sources.append(citation)
        
        return QueryResponse(
            success=True,
            question=request.question,
            answer=result["response"],
            sources=sources,
            multimodal_context_used=result.get("multimodal_context_used", False),
            image_analysis_count=result.get("image_analysis_count", 0),
            metadata={
                **result.get("metadata", {}),
                "multimodal_enabled": True,
                "query_type": "multimodal_rag",
                "model_used": query_params["multimodal_llm_model"],
                "image_context_included": request.include_image_context
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in multi-modal query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Multi-modal query failed: {str(e)}"
        )


@router.post(
    "/search/images",
    response_model=ImageSearchResponse,
    summary="Image-specific search",
    description="Search specifically within document images using visual analysis and OCR"
)
async def image_search(
    request: ImageSearchRequest,
    llamaindex: LlamaIndexService = Depends(get_llamaindex_service),
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> ImageSearchResponse:
    """
    Search specifically within document images.
    
    This endpoint focuses on:
    - Visual content analysis
    - OCR text extraction and search
    - Image similarity matching
    - Visual element detection
    """
    try:
        # Get configuration
        from app.config import get_settings
        settings = get_settings()
        multimodal_config = settings.get_multimodal_config()
        ocr_config = settings.get_ocr_config()
        
        # Validate image processing is enabled
        if not multimodal_config.get("image_processing_enabled", False):
            raise HTTPException(
                status_code=400,
                detail="Image processing is not enabled. Please check configuration."
            )
        
        # Build search parameters
        search_params = {
            "query": request.query,
            "search_type": request.search_type,
            "limit": request.limit,
            "similarity_threshold": request.similarity_threshold,
            "ocr_confidence_threshold": request.ocr_confidence_threshold,
            "visual_similarity_threshold": request.visual_similarity_threshold,
            "image_analysis_model": request.image_analysis_model or multimodal_config.get("image_analysis_model"),
            # Material-specific parameters
            "material_filters": request.material_filters,
            "material_types": request.material_types,
            "confidence_threshold": request.confidence_threshold,
            "spectral_filters": request.spectral_filters,
            "chemical_filters": request.chemical_filters,
            "mechanical_filters": request.mechanical_filters,
            "fusion_weights": request.fusion_weights,
            "enable_clip_embeddings": request.enable_clip_embeddings,
            "enable_llama_analysis": request.enable_llama_analysis,
            "include_analytics": request.include_analytics
        }
        
        # Filter documents if specified
        if request.document_ids:
            document_ids = request.document_ids
        else:
            # Get documents that have images
            documents = await supabase.list_documents(
                limit=1000,
                status_filter="completed"
            )
            # Filter to only documents with images
            document_ids = []
            for doc in documents.get("documents", []):
                # Check if document has associated images
                images = await supabase.get_document_images(doc["id"])
                if images:
                    document_ids.append(doc["id"])
        
        if not document_ids:
            return ImageSearchResponse(
                success=True,
                query=request.query,
                results=[],
                total_found=0,
                search_time_ms=0.0,
                analysis_depth="standard",
                ocr_enabled=ocr_config.get("ocr_enabled", False),
                metadata={
                    "searched_documents": 0,
                    "search_type": request.search_type,
                    "image_processing_enabled": True
                }
            )
        
        # Perform image search
        search_results = []
        for doc_id in document_ids:
            try:
                result = await llamaindex.image_search(
                    document_id=doc_id,
                    **search_params
                )
                
                if result["success"] and result.get("results"):
                    for item in result["results"]:
                        image_result = ImageSearchResult(
                            document_id=doc_id,
                            image_id=item.get("image_id", ""),
                            image_path=item.get("image_path", ""),
                            similarity_score=item.get("similarity_score", 0.0),
                            ocr_text=item.get("ocr_text", ""),
                            ocr_confidence=item.get("ocr_confidence", 0.0),
                            visual_description=item.get("visual_description", ""),
                            detected_elements=item.get("detected_elements", []),
                            page_number=item.get("page_number"),
                            metadata=item.get("metadata", {}),
                            # Enhanced material analysis fields
                            material_analysis=item.get("material_analysis", {}),
                            clip_embedding=item.get("clip_embedding", []),
                            llama_analysis=item.get("llama_analysis", {}),
                            material_type=item.get("material_type"),
                            material_confidence=item.get("material_confidence"),
                            spectral_properties=item.get("spectral_properties", {}),
                            chemical_composition=item.get("chemical_composition", {}),
                            mechanical_properties=item.get("mechanical_properties", {})
                        )
                        search_results.append(image_result)
                        
            except Exception as e:
                logger.warning(f"Error in image search for document {doc_id}: {e}")
                continue
        
        # Sort by similarity score
        search_results.sort(key=lambda x: x.similarity_score, reverse=True)
        limited_results = search_results[:request.limit]

        # Calculate search time (mock for now)
        search_time_ms = 150.0

        return ImageSearchResponse(
            success=True,
            query=request.query,
            results=limited_results,
            total_found=len(search_results),
            search_time_ms=search_time_ms,
            analysis_depth="standard",
            ocr_enabled=ocr_config.get("ocr_enabled", False),
            metadata={
                "searched_documents": len(document_ids),
                "search_type": request.search_type,
                "image_processing_enabled": True,
                "returned_results": len(limited_results)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in image search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Image search failed: {str(e)}"
        )


@router.post(
    "/analyze/multimodal",
    response_model=MultiModalAnalysisResponse,
    summary="Comprehensive multi-modal document analysis",
    description="Perform comprehensive analysis of document content including text, images, and structure"
)
async def multimodal_analysis(
    request: MultiModalAnalysisRequest,
    llamaindex: LlamaIndexService = Depends(get_llamaindex_service),
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> MultiModalAnalysisResponse:
    """
    Perform comprehensive multi-modal analysis of a document.
    
    This endpoint provides:
    - Complete document structure analysis
    - Text content analysis and summarization
    - Image content analysis and description
    - OCR text extraction and analysis
    - Cross-modal relationship detection
    """
    try:
        # Verify document exists
        document_data = await supabase.get_document_by_id(request.document_id)
        if not document_data:
            raise HTTPException(
                status_code=404,
                detail=f"Document {request.document_id} not found"
            )
        
        # Get configuration
        from app.config import get_settings
        settings = get_settings()
        multimodal_config = settings.get_multimodal_config()
        
        # Validate multi-modal is enabled
        if not multimodal_config.get("enable_multimodal", False):
            raise HTTPException(
                status_code=400,
                detail="Multi-modal analysis is not enabled. Please check configuration."
            )
        
        # Build analysis parameters
        analysis_params = {
            "document_id": request.document_id,
            "analysis_types": request.analysis_types,
            "include_text_analysis": request.include_text_analysis,
            "include_image_analysis": request.include_image_analysis,
            "include_ocr_analysis": request.include_ocr_analysis,
            "include_structure_analysis": request.include_structure_analysis,
            "analysis_depth": request.analysis_depth,
            "multimodal_llm_model": request.multimodal_llm_model or multimodal_config.get("multimodal_llm_model")
        }
        
        # Perform comprehensive analysis
        result = await llamaindex.multimodal_analysis(**analysis_params)
        
        if not result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Multi-modal analysis failed: {result.get('error', 'Unknown error')}"
            )
        
        return MultiModalAnalysisResponse(
            success=True,
            document_id=request.document_id,
            analysis_types=request.analysis_types,
            text_analysis=result.get("text_analysis", {}),
            image_analysis=result.get("image_analysis", {}),
            ocr_analysis=result.get("ocr_analysis", {}),
            structure_analysis=result.get("structure_analysis", {}),
            cross_modal_insights=result.get("cross_modal_insights", {}),
            metadata={
                **result.get("metadata", {}),
                "document_title": document_data.get("title", ""),
                "analysis_depth": request.analysis_depth,
                "model_used": analysis_params["multimodal_llm_model"],
                "multimodal_enabled": True
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in multi-modal analysis for document {request.document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Multi-modal analysis failed: {str(e)}"
        )


# ============================================================================
# MATERIAL-SPECIFIC VISUAL SEARCH ENDPOINTS
# ============================================================================

async def get_material_visual_search_service() -> MaterialVisualSearchService:
    """Dependency to get Material Visual Search service instance."""
    try:
        # Try to initialize the service
        service = MaterialVisualSearchService()
        return service
    except Exception as e:
        logger.error(f"Failed to initialize Material Visual Search service: {e}")
        # Return a fallback service instead of raising an exception
        return MaterialVisualSearchService(config={
            "supabase_url": "",
            "supabase_service_key": "",
            "enable_fallback": True
        })


@router.post(
    "/search/materials/visual",
    response_model=MaterialSearchResponse,
    summary="Material-specific visual search",
    description="Perform visual search with material property filtering and analysis"
)
async def material_visual_search(
    request: MaterialSearchRequest,
    material_search_service: MaterialVisualSearchService = Depends(get_material_visual_search_service)
) -> MaterialSearchResponse:
    """
    Perform material-specific visual search with advanced filtering and analysis.
    
    This endpoint provides:
    - Visual similarity search using CLIP embeddings
    - Material property filtering (spectral, chemical, mechanical, thermal)
    - LLaMA Vision analysis for material understanding
    - Multi-modal fusion with configurable weights
    - Integration with Supabase visual search infrastructure
    """
    try:
        logger.info(f"Material visual search requested: {request.search_type}")
        
        # Execute material visual search
        result = await material_search_service.search_materials(request)
        
        if not result.success:
            raise HTTPException(
                status_code=400,
                detail="Material visual search failed"
            )
        
        logger.info(f"Material search completed: {result.total_results} results")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in material visual search: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Material visual search failed: {str(e)}"
        )


@router.post(
    "/analyze/materials/image",
    response_model=SuccessResponse,
    summary="Analyze material image",
    description="Analyze a material image using integrated visual analysis"
)
async def analyze_material_image(
    request: Dict[str, Any],
    material_search_service: MaterialVisualSearchService = Depends(get_material_visual_search_service)
) -> SuccessResponse:
    """
    Analyze a material image using integrated visual analysis.
    
    This endpoint provides comprehensive material analysis including:
    - Visual feature extraction
    - Material identification and classification
    - Spectral, chemical, and mechanical property analysis
    - CLIP embedding generation
    - LLaMA Vision material understanding
    """
    try:
        image_data = request.get("image_data")
        analysis_types = request.get("analysis_types", ["visual", "spectral", "chemical"])
        
        if not image_data:
            raise HTTPException(
                status_code=400,
                detail="image_data is required"
            )
        
        # Analyze material image
        result = await material_search_service.analyze_material_image(
            image_data=image_data,
            analysis_types=analysis_types
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=f"Material image analysis failed: {result.get('error', 'Unknown error')}"
            )
        
        return SuccessResponse(
            success=True,
            message="Material image analysis completed",
            data=result.get("analysis", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in material image analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Material image analysis failed: {str(e)}"
        )


@router.post(
    "/embeddings/materials/generate",
    response_model=SuccessResponse,
    summary="Generate material embeddings",
    description="Generate CLIP and custom embeddings for material images"
)
async def generate_material_embeddings(
    request: Dict[str, Any],
    material_search_service: MaterialVisualSearchService = Depends(get_material_visual_search_service)
) -> SuccessResponse:
    """
    Generate embeddings for material images.
    
    This endpoint provides:
    - CLIP embedding generation for visual similarity
    - Custom material-specific embeddings
    - Batch processing for multiple images
    - Embedding metadata and quality metrics
    """
    try:
        image_data = request.get("image_data")
        embedding_types = request.get("embedding_types", ["clip"])
        
        if not image_data:
            raise HTTPException(
                status_code=400,
                detail="image_data is required"
            )
        
        # Generate embeddings
        result = await material_search_service.generate_material_embeddings(
            image_data=image_data,
            embedding_types=embedding_types
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=f"Embedding generation failed: {result.get('error', 'Unknown error')}"
            )
        
        return SuccessResponse(
            success=True,
            message="Material embeddings generated successfully",
            data={
                "embeddings": result.get("embeddings", {}),
                "metadata": result.get("embedding_metadata", {})
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in material embedding generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Material embedding generation failed: {str(e)}"
        )


@router.get(
    "/search/materials/{material_id}/similar",
    response_model=SuccessResponse,
    summary="Find similar materials",
    description="Find materials similar to a reference material using visual and property analysis"
)
async def find_similar_materials(
    material_id: str,
    similarity_threshold: float = Query(0.75, ge=0.0, le=1.0, description="Minimum similarity threshold"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results"),
    material_search_service: MaterialVisualSearchService = Depends(get_material_visual_search_service)
) -> SuccessResponse:
    """
    Find materials similar to a reference material.
    
    This endpoint performs:
    - Visual similarity analysis using CLIP embeddings
    - Material property comparison
    - Multi-modal similarity scoring
    - Ranked results with confidence scores
    """
    try:
        # Search for similar materials
        result = await material_search_service.search_similar_materials(
            reference_material_id=material_id,
            similarity_threshold=similarity_threshold,
            limit=limit
        )
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=f"Similar material search failed: {result.get('error', 'Unknown error')}"
            )
        
        return SuccessResponse(
            success=True,
            message=f"Found {result.get('total_found', 0)} similar materials",
            data={
                "reference_material_id": material_id,
                "similar_materials": result.get("similar_materials", []),
                "total_found": result.get("total_found", 0),
                "search_metadata": result.get("search_metadata", {})
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar materials for {material_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Similar material search failed: {str(e)}"
        )


@router.get(
    "/search/materials/health",
    response_model=SuccessResponse,
    summary="Material visual search health check",
    description="Check health and availability of material visual search services"
)
async def material_search_health_check(
    material_search_service: MaterialVisualSearchService = Depends(get_material_visual_search_service)
) -> SuccessResponse:
    """
    Check the health of material visual search services.
    
    This endpoint provides information about:
    - Material Visual Search service status
    - Supabase visual search function connectivity
    - Material Kai service integration status
    - CLIP embedding service availability
    - Overall system health
    """
    try:
        health_status = await material_search_service.health_check()
        
        return SuccessResponse(
            success=True,
            message="Material visual search health check completed",
            data=health_status
        )
        
    except Exception as e:
        logger.error(f"Error in material search health check: {e}")
        return SuccessResponse(
            success=False,
            message="Material visual search health check failed",
            data={
                "error": str(e),
                "service": "material_visual_search",
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat()
            }
        )