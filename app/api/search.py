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
    DocumentQueryRequest,
    DocumentQueryResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SimilaritySearchRequest,
    SimilaritySearchResponse,
    RelatedDocumentsRequest,
    RelatedDocumentsResponse,
    DocumentSummaryRequest,
    DocumentSummaryResponse,
    EntityExtractionRequest,
    EntityExtractionResponse,
    DocumentComparisonRequest,
    DocumentComparisonResponse
)
from ..schemas.common import ErrorResponse, SuccessResponse
from ..services.llamaindex_service import LlamaIndexService
from ..services.supabase_client import SupabaseClient

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["search", "rag"])

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
            query=request.text,
            document_ids=request.document_ids,
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
            text=request.text,
            similar_documents=similar_documents,
            total_found=semantic_result.total_results,
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