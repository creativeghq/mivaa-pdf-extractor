"""
Search and RAG API Endpoints

This module provides advanced search and RAG (Retrieval-Augmented Generation)
endpoints using Claude 4.5 + Direct Vector DB for semantic search, document querying,
and intelligent document analysis.
"""

import logging
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
from ..services.search.rag_service import RAGService
from ..services.core.supabase_client import SupabaseClient
from ..services.search.material_visual_search_service import (
    MaterialVisualSearchService,
    MaterialSearchRequest,
    MaterialSearchResponse,
    get_material_visual_search_service
)

# Import unified search service (Step 7)
from ..services.search.unified_search_service import (
    UnifiedSearchService,
    SearchConfig,
    SearchStrategy
)

# Import centralized dependencies
from ..dependencies import get_rag_service, get_supabase_client

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api", tags=["Search", "Embeddings", "Chat"])

@router.post(
    "/search/semantic",
    response_model=SemanticSearchResponse,
    summary="Semantic search across documents",
    description="Perform semantic search across multiple documents using vector similarity"
)
async def semantic_search(
    request: SemanticSearchRequest,
    rag: RAGService = Depends(get_rag_service),
    supabase: SupabaseClient = Depends(get_supabase_client)
) -> SemanticSearchResponse:
    """Semantic search across `document_chunks` (1024D Voyage halfvec).

    Distance is computed in Postgres via the `match_document_chunks_semantic`
    RPC — the previous implementation pulled every embedding into Python and
    computed cosine in numpy with a hardcoded 1536D check, which silently
    returned zero results for our 1024D vectors.
    """
    import asyncio

    try:
        # Generate query embedding (Voyage 1024D, query input_type)
        try:
            embedding_result = await rag.embedding_service.generate_embedding(request.query)
            if not embedding_result or not embedding_result.embedding:
                raise Exception("Failed to generate query embedding")
            query_embedding = embedding_result.embedding
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate query embedding: {e}"
            )

        # Resolve document filter
        if request.document_ids:
            document_ids = request.document_ids
        else:
            documents = await supabase.list_documents(
                limit=1000,
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
                    "similarity_threshold": request.similarity_threshold,
                },
            )

        # Postgres-side cosine search — single round trip, halfvec_cosine_ops
        # index handles ordering. NULL embeddings filtered inside the RPC.
        rpc_result = await asyncio.to_thread(
            lambda: supabase.client.rpc(
                "match_document_chunks_semantic",
                {
                    "query_embedding": query_embedding,
                    "similarity_threshold": float(request.similarity_threshold),
                    "match_count": int(request.max_results),
                    "filter_document_ids": document_ids,
                    "filter_workspace_id": None,
                },
            ).execute()
        )

        rows = rpc_result.data or []
        search_results = [
            {
                "document_id": row.get("document_id", ""),
                "score": float(row.get("similarity") or 0.0),
                "content": row.get("content", ""),
                "metadata": row.get("metadata") or {},
            }
            for row in rows
        ]
        limited_results = search_results  # RPC already applies LIMIT

        # Enrich results with document metadata
        enriched_results = []
        for result in limited_results:
            document_id = result.get("document_id")

            # Get document metadata from database
            try:
                doc_result = await asyncio.to_thread(
                    lambda: supabase.client.table("documents")
                    .select("filename, content_type, metadata, created_at, processing_status")
                    .eq("id", document_id)
                    .single()
                    .execute()
                )

                if doc_result.data:
                    doc = doc_result.data
                    # Get document name from multiple possible sources
                    document_name = (
                        doc.get("filename") or  # Primary: filename field
                        doc.get("metadata", {}).get("filename") or  # Secondary: filename in metadata
                        doc.get("metadata", {}).get("title") or     # Tertiary: title in metadata
                        f"Document {document_id[:8]}"  # Fallback: use document ID
                    )

                    enriched_result = {
                        **result,
                        "document_name": document_name,
                        "filename": doc.get("filename"),
                        "content_type": doc.get("content_type"),
                        "created_at": doc.get("created_at"),
                        "processing_status": doc.get("processing_status"),
                        "source_metadata": {
                            "filename": doc.get("filename"),
                            "content_type": doc.get("content_type"),
                            "file_size": doc.get("metadata", {}).get("file_size"),
                            "page_count": doc.get("metadata", {}).get("page_count"),
                            **doc.get("metadata", {})
                        }
                    }
                    enriched_results.append(enriched_result)
                else:
                    # If document not found, use original result with placeholder
                    enriched_result = {
                        **result,
                        "document_name": f"Document {document_id[:8]}",
                        "filename": "Unknown",
                        "content_type": "application/pdf",
                        "source_metadata": {"error": "Document metadata not found"}
                    }
                    enriched_results.append(enriched_result)

            except Exception as e:
                logger.warning(f"Failed to get metadata for document {document_id}: {e}")
                # Use original result with error indication
                enriched_result = {
                    **result,
                    "document_name": f"Document {document_id[:8]}",
                    "filename": "Error loading metadata",
                    "content_type": "application/pdf",
                    "source_metadata": {"error": str(e)}
                }
                enriched_results.append(enriched_result)

        return SemanticSearchResponse(
            success=True,
            query=request.query,
            results=enriched_results,
            total_results=len(search_results),
            metadata={
                "searched_documents": len(document_ids),
                "similarity_threshold": request.similarity_threshold,
                "returned_results": len(enriched_results),
                "metadata_enriched": True
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
    rag: RAGService = Depends(get_rag_service),
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
            max_results=request.limit,
            similarity_threshold=request.similarity_threshold
        )
        
        semantic_result = await semantic_search(semantic_request, rag, supabase)
        
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
    "/search/health",
    response_model=SuccessResponse,
    summary="Search service health check",
    description="Check the health and availability of search and RAG services"
)
async def search_health_check(
    rag: RAGService = Depends(get_rag_service)
) -> SuccessResponse:
    """
    Check the health of search and RAG services.
    
    This endpoint provides information about the availability and status
    of the RAG service and its components.
    """
    try:
        health_status = await rag.health_check()
        
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
    rag: RAGService = Depends(get_rag_service),
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
                # Use enhanced RAG service for multi-modal search
                result = await rag.multimodal_search(
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

        # Enrich results with document metadata
        enriched_results = []
        for result in limited_results:
            document_id = result.document_id

            # Get document metadata from database
            try:
                doc_result = await asyncio.to_thread(
                    lambda: supabase.client.table("documents")
                    .select("filename, content_type, metadata, created_at, processing_status")
                    .eq("id", document_id)
                    .single()
                    .execute()
                )

                if doc_result.data:
                    doc = doc_result.data
                    # Get document name from multiple possible sources
                    document_name = (
                        doc.get("filename") or  # Primary: filename field
                        doc.get("metadata", {}).get("filename") or  # Secondary: filename in metadata
                        doc.get("metadata", {}).get("title") or     # Tertiary: title in metadata
                        f"Document {document_id[:8]}"  # Fallback: use document ID
                    )

                    # Update result with metadata
                    result.document_name = document_name
                    result.filename = doc.get("filename")
                    result.processing_status = doc.get("processing_status")
                    result.created_at = doc.get("created_at")
                    result.source_metadata = {
                        "filename": doc.get("filename"),
                        "content_type": doc.get("content_type"),
                        "file_size": doc.get("metadata", {}).get("file_size"),
                        "page_count": doc.get("metadata", {}).get("page_count"),
                        **doc.get("metadata", {})
                    }

                enriched_results.append(result)

            except Exception as e:
                logger.warning(f"Failed to get metadata for document {document_id}: {e}")
                # Use original result with error indication
                result.document_name = f"Document {document_id[:8]}"
                result.filename = "Error loading metadata"
                result.source_metadata = {"error": str(e)}
                enriched_results.append(result)

        return SearchResponse(
            success=True,
            query=request.query,
            results=enriched_results,
            total_found=len(search_results),
            search_time_ms=200.0,
            search_type="multimodal",
            filters_applied={},
            metadata={
                "searched_documents": len(document_ids),
                "multimodal_enabled": True,
                "search_type": "multimodal",
                "returned_results": len(enriched_results),
                "ocr_enabled": ocr_config.get("ocr_enabled", False),
                "image_analysis_enabled": request.include_images,
                "metadata_enriched": True
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
    rag: RAGService = Depends(get_rag_service),
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
        if request.context_documents:
            document_ids = request.context_documents
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
        # Note: Using multimodal_analysis as a fallback since multimodal_query method signature doesn't match
        try:
            if document_ids:
                # Use the first document for analysis
                result = await rag.multimodal_analysis(
                    document_id=document_ids[0],
                    analysis_types=["text", "image", "ocr"],
                    include_text_analysis=True,
                    include_image_analysis=request.include_image_context,
                    include_ocr_analysis=True,
                    include_structure_analysis=True,
                    analysis_depth=request.image_analysis_depth,
                    multimodal_llm_model=query_params["multimodal_llm_model"]
                )
            else:
                result = {
                    "success": False,
                    "error": "No documents available for analysis"
                }
        except Exception as e:
            logger.error(f"Multimodal analysis failed: {e}")
            result = {
                "success": False,
                "error": f"Analysis failed: {str(e)}"
            }

        if not result.get("success", False):
            raise HTTPException(
                status_code=400,
                detail=f"Multi-modal query failed: {result.get('error', 'Unknown error')}"
            )
        
        # Build source citations with multi-modal information
        sources = []
        analysis_results = result.get("analysis_results", {})

        # Create a source citation from the analysis results
        if document_ids:
            citation = SourceCitation(
                document_id=document_ids[0],
                content_excerpt=analysis_results.get("text_summary", "")[:200] + "..." if analysis_results.get("text_summary") else "",
                content_type="multimodal",
                ocr_excerpt=analysis_results.get("ocr_summary", "")[:200] + "..." if analysis_results.get("ocr_summary") else "",
                image_reference="",
                multimodal_confidence=analysis_results.get("confidence_score", 0.0),
                page_number=None,
                metadata=result.get("metadata", {})
            )
            sources.append(citation)

        # Generate a response based on the analysis
        answer = f"Based on the multimodal analysis: {analysis_results.get('summary', 'Analysis completed successfully.')}"
        if analysis_results.get("text_summary"):
            answer += f"\n\nText content: {analysis_results['text_summary']}"
        if analysis_results.get("image_summary") and request.include_image_context:
            answer += f"\n\nImage content: {analysis_results['image_summary']}"

        return QueryResponse(
            success=True,
            question=request.question,
            answer=answer,
            sources=sources,
            multimodal_context_used=request.include_image_context,
            image_analysis_count=analysis_results.get("image_count", 0),
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
    rag: RAGService = Depends(get_rag_service),
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
        import time
        search_start_time = time.time()

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
            "enable_vision_analysis": request.enable_vision_analysis,
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
                result = await rag.image_search(
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
                            vision_analysis=item.get("vision_analysis", {}),
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

        # Calculate actual search time
        search_time_ms = (time.time() - search_start_time) * 1000

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
    rag: RAGService = Depends(get_rag_service),
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
        result = await rag.multimodal_analysis(**analysis_params)
        
        if not result["success"]:
            raise HTTPException(
                status_code=400,
                detail=f"Multi-modal analysis failed: {result.get('error', 'Unknown error')}"
            )
        
        return MultiModalAnalysisResponse(
            success=True,
            document_id=request.document_id,
            document_name=document_data.get("data", {}).get("title", f"Document {request.document_id}"),
            analysis_types=request.analysis_types,
            analysis_time_ms=result.get("metadata", {}).get("processing_time_ms", 1500),
            total_pages=document_data.get("data", {}).get("page_count", 1),
            total_images=document_data.get("data", {}).get("image_count", 0),
            total_text_chunks=document_data.get("data", {}).get("chunk_count", 5),
            text_analysis=result.get("text_analysis", {}),
            image_analysis=result.get("image_analysis", {}),
            ocr_analysis=result.get("ocr_analysis", {}),
            structure_analysis=result.get("structure_analysis", {}),
            cross_modal_insights=result.get("cross_modal_insights", {}),
            metadata={
                **result.get("metadata", {}),
                "document_title": document_data.get("data", {}).get("title", ""),
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
    - Visual similarity search using SLIG embeddings
    - Material property filtering (spectral, chemical, mechanical, thermal)
    - Qwen Vision analysis for material understanding
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
    - SLIG embedding generation
    - Qwen Vision material understanding
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
    - Visual similarity analysis using SLIG embeddings
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
    - SLIG embedding service availability
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
            message="Material visual search health check failed"
        )


# ============================================================================
# PER-ASPECT SEARCH ENDPOINTS
#
# Pre-2026-05-04 these called _generate_specialized_clip_embeddings (which
# never existed in the SLIG codebase — the endpoints had been silently 4xx-ing
# on AttributeError since the 2026-04 CLIP→SLIG migration). They also
# referenced ImageSearchRequest fields (query_image, workspace_id,
# min_similarity) that don't exist on the schema, so even if the embedding
# method were rewired the request parsing would fail.
#
# Rewritten 2026-05-05 to use the v2 aspect architecture:
#   - Voyage `voyage-3` 1024D queries (matches the aspect collection space)
#   - Optional query_image path runs Claude Opus 4.7 vision_analysis on the
#     image, then the per-aspect serializer to build a query string Voyage-
#     embeds — exactly mirrors the ingestion pipeline so the query embedding
#     is a sibling of the row embeddings it'll be compared against.
#   - Optional query_text path skips the Opus pass and Voyage-embeds the
#     user's text directly. Cheaper (no vision call), good for "all images
#     that look like 'warm white veined marble'" style queries.
# ============================================================================


class AspectSearchRequest(BaseModel):
    """Request schema for the four /search/by-<aspect> endpoints.

    Either `query_image` or `query_text` must be provided. If both are given,
    image takes precedence (the user obviously wants per-aspect retrieval
    grounded in actual material visible in the image).
    """

    query_image: Optional[str] = Field(
        None,
        description=(
            "Base64-encoded image OR HTTPS URL. When provided, the server runs "
            "Claude Opus 4.7 vision_analysis on the image and Voyage-embeds the "
            "per-aspect text derived from the result — same pipeline as ingestion."
        ),
    )
    query_text: Optional[str] = Field(
        None,
        description=(
            "Free-form text describing the aspect to match (e.g. 'warm white "
            "veined marble' for color search). Voyage-embedded directly. "
            "Cheaper than query_image; use when the user already knows the words."
        ),
    )

    workspace_id: Optional[str] = Field(None, description="Restrict results to this workspace")
    document_id: Optional[str] = Field(None, description="Restrict results to this document")
    limit: int = Field(10, ge=1, le=50, description="Max results to return")
    min_similarity: float = Field(0.5, ge=0.0, le=1.0, description="Drop results below this cosine similarity")

    class Config:
        json_schema_extra = {
            "example": {
                "query_text": "warm white veined marble",
                "limit": 15,
                "min_similarity": 0.6,
            }
        }


class AspectSearchResultItem(BaseModel):
    image_id: str
    similarity_score: float
    distance: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class AspectSearchResponse(BaseModel):
    success: bool
    aspect: str
    results: List[AspectSearchResultItem]
    total_results: int
    query_summary: Optional[str] = Field(
        None,
        description=(
            "The text string the server actually Voyage-embedded for the query. "
            "Surfaced so callers can see exactly what was compared — particularly "
            "useful when query_image was provided (shows the Opus output)."
        ),
    )


async def _build_aspect_query_embedding(
    aspect: str,
    query_image: Optional[str],
    query_text: Optional[str],
) -> tuple[Optional[List[float]], Optional[str], Optional[str]]:
    """Resolve `(query_image | query_text) → (1024D Voyage embedding, source text, error)`.

    The single chokepoint for all 4 aspect endpoints. Returns:
        (embedding, source_text, error) — exactly one of (embedding, error)
        is populated. `source_text` is the string Voyage embedded, returned
        to the caller for transparency.

    `aspect` is one of color/texture/style/material. Drives which serializer
    we call when query_image was provided.
    """
    if not query_image and not query_text:
        return None, None, "Provide either query_image or query_text"

    from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
    voyage_svc = RealEmbeddingsService()

    # Path A — query_image: run Opus vision_analysis, build aspect text via
    # serializer, embed. Mirrors ingestion exactly so query and row vectors
    # live in the same Voyage embedding space.
    if query_image:
        from app.models.vision_analysis import (
            VisionAnalysis,
            VISION_ANALYSIS_TOOL,
            ASPECT_SERIALIZERS,
        )
        import base64 as _b64
        import os as _os

        anthropic_key = _os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_key:
            return None, None, "ANTHROPIC_API_KEY not configured"

        # Accept either a data URL or a raw base64 string. URLs are fetched
        # before sending to Anthropic so we don't have to deal with anthropic
        # supporting only certain URL patterns.
        if query_image.startswith("http"):
            import httpx
            try:
                async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                    resp = await client.get(query_image)
                    if resp.status_code != 200:
                        return None, None, f"Failed to fetch query_image (HTTP {resp.status_code})"
                    image_b64 = _b64.b64encode(resp.content).decode()
            except Exception as e:
                return None, None, f"Failed to fetch query_image: {e}"
        elif query_image.startswith("data:"):
            try:
                _, _, encoded = query_image.partition("base64,")
                if not encoded:
                    return None, None, "Malformed data URL (no base64 payload)"
                image_b64 = encoded
            except Exception as e:
                return None, None, f"Failed to parse data URL: {e}"
        else:
            image_b64 = query_image  # assume raw base64

        # Run vision_analysis via Anthropic tool use. Schema-locked output.
        try:
            import httpx
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": anthropic_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-opus-4-7",
                        "max_tokens": 4096,
                        "tools": [VISION_ANALYSIS_TOOL],
                        "tool_choice": {"type": "tool", "name": "emit_vision_analysis"},
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "image", "source": {
                                    "type": "base64", "media_type": "image/jpeg", "data": image_b64,
                                }},
                                {"type": "text", "text": (
                                    "Use the emit_vision_analysis tool to return a "
                                    "structured catalog-grade material analysis for this image."
                                )},
                            ],
                        }],
                    },
                )
                if resp.status_code != 200:
                    return None, None, f"Anthropic vision_analysis returned HTTP {resp.status_code}"
                payload = resp.json()
                tool_block = next(
                    (b for b in payload.get("content", []) if b.get("type") == "tool_use"),
                    None,
                )
                if not tool_block:
                    return None, None, "Anthropic response missing tool_use block"
                va = VisionAnalysis(**tool_block["input"])
        except Exception as e:
            return None, None, f"Anthropic vision_analysis call failed: {e}"

        serializer = ASPECT_SERIALIZERS.get(aspect)
        if not serializer:
            return None, None, f"Unknown aspect: {aspect}"
        source_text = serializer(va)
        if not source_text:
            return None, None, f"VisionAnalysis from query_image had no {aspect} content (e.g. empty colors[])"
    else:
        # Path B — query_text: skip Opus, embed the user's text directly.
        source_text = query_text.strip()
        if not source_text:
            return None, None, "query_text is empty"

    # Common terminus: Voyage-embed the source_text at 1024D.
    try:
        vec = await voyage_svc._generate_text_embedding(text=source_text, input_type="query")
    except Exception as e:
        return None, None, f"Voyage embed failed: {e}"

    if not vec or len(vec) != 1024:
        return None, None, f"Voyage returned wrong-dim embedding (len={len(vec) if vec else 0}, expected 1024)"

    return vec, source_text, None


async def _run_aspect_search(
    aspect: str,
    request: AspectSearchRequest,
) -> AspectSearchResponse:
    """Shared body for the four /search/by-<aspect> endpoints."""
    from app.services.embeddings.vecs_service import get_vecs_service

    embedding, source_text, error = await _build_aspect_query_embedding(
        aspect=aspect,
        query_image=request.query_image,
        query_text=request.query_text,
    )
    if error:
        raise HTTPException(status_code=400, detail=error)

    vecs_svc = get_vecs_service()
    raw_results = await vecs_svc.search_specialized_embeddings(
        query_embedding=embedding,
        embedding_type=aspect,
        limit=request.limit,
        workspace_id=request.workspace_id,
        document_id=request.document_id,
        include_metadata=True,
    )

    # Apply min_similarity client-side — search_specialized_embeddings doesn't
    # accept a threshold (it converts distance → similarity inside, but doesn't
    # filter). Cosine similarity sits in [-1, 1]; the cutoff is requested by
    # the caller and we trust their value.
    filtered = [
        AspectSearchResultItem(
            image_id=r.get("image_id"),
            similarity_score=r.get("similarity_score"),
            distance=r.get("distance"),
            metadata=r.get("metadata"),
        )
        for r in raw_results
        if (r.get("similarity_score") or 0.0) >= request.min_similarity
    ]

    logger.info(
        f"✅ /search/by-{aspect}: {len(filtered)}/{len(raw_results)} results above min_similarity "
        f"(query_path={'image' if request.query_image else 'text'})"
    )

    return AspectSearchResponse(
        success=True,
        aspect=aspect,
        results=filtered,
        total_results=len(filtered),
        query_summary=source_text,
    )


@router.post(
    "/search/by-color",
    response_model=AspectSearchResponse,
    summary="Search images by color palette",
    description=(
        "Per-aspect search against `image_color_embeddings`. With query_image, "
        "the server runs vision_analysis and Voyage-embeds the resulting "
        "VisionAnalysis.colors[] string. With query_text, it Voyage-embeds your "
        "text directly. Either way the query lives in the same Voyage 1024D "
        "space as the rows it's compared against."
    ),
)
async def search_by_color(request: AspectSearchRequest) -> AspectSearchResponse:
    return await _run_aspect_search("color", request)


@router.post(
    "/search/by-texture",
    response_model=AspectSearchResponse,
    summary="Search images by texture pattern",
    description=(
        "Per-aspect search against `image_texture_embeddings`. Source field set: "
        "VisionAnalysis.textures[] + finish. See /search/by-color for the query "
        "model — same shape across the four aspect endpoints."
    ),
)
async def search_by_texture(request: AspectSearchRequest) -> AspectSearchResponse:
    return await _run_aspect_search("texture", request)


@router.post(
    "/search/by-style",
    response_model=AspectSearchResponse,
    summary="Search images by design style",
    description=(
        "Per-aspect search against `image_style_embeddings`. Source field set: "
        "VisionAnalysis.style + surface_pattern + applications."
    ),
)
async def search_by_style(request: AspectSearchRequest) -> AspectSearchResponse:
    return await _run_aspect_search("style", request)


@router.post(
    "/search/by-material",
    response_model=AspectSearchResponse,
    summary="Search images by material type",
    description=(
        "Per-aspect search against `image_material_embeddings`. Source field set: "
        "VisionAnalysis.material_type + category + subcategory."
    ),
)
async def search_by_material(request: AspectSearchRequest) -> AspectSearchResponse:
    return await _run_aspect_search("material", request)


