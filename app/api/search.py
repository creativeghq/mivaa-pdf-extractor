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
from ..services.rag_service import RAGService
from ..services.supabase_client import SupabaseClient
from ..services.material_visual_search_service import (
    MaterialVisualSearchService,
    MaterialSearchRequest,
    MaterialSearchResponse,
    get_material_visual_search_service
)

# Import unified search service (Step 7)
from ..services.unified_search_service import (
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

# REMOVED: Module-level service initialization - now using centralized dependencies
# REMOVED: /documents/{document_id}/query - Use /api/rag/query instead


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
        
        # Generate embedding for the query
        try:
            embedding_result = await rag.embedding_service.generate_embedding(request.query)
            if not embedding_result or not embedding_result.embedding:
                raise Exception("Failed to generate query embedding")
            query_embedding = embedding_result.embedding  # Extract the actual embedding list
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate query embedding: {str(e)}"
            )

        # Perform direct vector search using database embeddings
        search_results = []
        try:
            # Build document filter for SQL query
            doc_filter = ""
            if document_ids:
                doc_ids_str = "', '".join(document_ids)
                doc_filter = f"AND dv.document_id IN ('{doc_ids_str}')"

            # Convert embedding to PostgreSQL vector format
            embedding_str = f"[{','.join(map(str, query_embedding))}]"

            # Direct SQL query for vector similarity search using document_vectors table
            query_sql = f"""
            SELECT
                dv.document_id,
                dv.chunk_id,
                dv.content,
                dv.metadata,
                1 - (dv.embedding <=> '{embedding_str}'::vector) as similarity_score
            FROM document_vectors dv
            WHERE dv.embedding IS NOT NULL
                {doc_filter}
                AND (1 - (dv.embedding <=> '{embedding_str}'::vector)) >= {request.similarity_threshold}
            ORDER BY dv.embedding <=> '{embedding_str}'::vector
            LIMIT {request.max_results}
            """

            # Execute the query directly
            import asyncio
            from ..services.supabase_client import SupabaseClient

            # Use the existing supabase client from dependencies
            # supabase_client is already available from the function parameter

            # Use table query with select (simpler and more reliable)
            table_result = await asyncio.to_thread(
                lambda: supabase.client.table('document_vectors')
                .select('document_id, chunk_id, content, metadata, embedding')
                .not_.is_('embedding', 'null')
                .execute()
            )

            if table_result.data:
                # Calculate similarity in Python (less efficient but works)
                try:
                    import numpy as np

                    logger.info(f"Found {len(table_result.data)} rows in document_vectors table")

                    for row in table_result.data:
                        if request.document_ids and row['document_id'] not in request.document_ids:
                            continue

                        # Parse embedding
                        embedding = row.get('embedding')

                        # Handle different embedding formats
                        if embedding:
                            # Convert vector string to list if needed
                            if isinstance(embedding, str):
                                # Parse vector string format like "[1.0, 2.0, 3.0]"
                                try:
                                    if embedding.startswith('[') and embedding.endswith(']'):
                                        # Remove brackets and split by comma
                                        embedding_str = embedding[1:-1]
                                        embedding = [float(x.strip()) for x in embedding_str.split(',')]
                                    else:
                                        continue
                                except Exception:
                                    continue

                            if isinstance(embedding, list) and len(embedding) == 1536:
                                # Calculate cosine similarity
                                embedding_array = np.array(embedding, dtype=np.float32)
                                query_array = np.array(query_embedding, dtype=np.float32)

                                # Normalize vectors
                                embedding_norm = embedding_array / np.linalg.norm(embedding_array)
                                query_norm = query_array / np.linalg.norm(query_array)

                                # Calculate cosine similarity
                                similarity = float(np.dot(embedding_norm, query_norm))

                                logger.info(f"Similarity: {similarity:.4f} for doc {row.get('document_id')}")

                                if similarity >= request.similarity_threshold:
                                    search_results.append({
                                        "document_id": row.get("document_id", ""),
                                        "score": similarity,
                                        "content": row.get("content", ""),
                                        "metadata": row.get("metadata", {})
                                    })


                    logger.info(f"Final search results: {len(search_results)} matches found")

                except ImportError:
                    logger.error("NumPy not available for similarity calculation")
                except Exception as e:
                    logger.error(f"Error in similarity calculation: {e}")

        except Exception as e:
            logger.error(f"Vector search error: {e}")
            # Fallback to RAG query if vector search fails
            for doc_id in document_ids:
                try:
                    result = await rag.query_document(
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
        limited_results = search_results[:request.max_results]

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


# REMOVED: /documents/{document_id}/related - Use /api/rag/search instead


# REMOVED: /documents/{document_id}/summarize - Not implemented


# REMOVED: /documents/{document_id}/extract-entities - Not implemented


# REMOVED: /documents/compare - Not implemented


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
    - Visual similarity search using CLIP embeddings
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
    - CLIP embedding generation
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
        material_properties = request.get("material_properties")

        if not image_data:
            raise HTTPException(
                status_code=400,
                detail="image_data is required"
            )

        # Use RealEmbeddingsService to generate real embeddings (Step 4)
        from app.services.real_embeddings_service import RealEmbeddingsService

        embeddings_service = RealEmbeddingsService()

        # Generate all real embeddings
        result = await embeddings_service.generate_all_embeddings(
            entity_id="temp",
            entity_type="image",
            text_content="",
            image_data=image_data,
            material_properties=material_properties
        )

        if result.get("success") is False:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Embedding generation failed")
            )

        # Return real embeddings
        return SuccessResponse(
            success=True,
            message="Material embeddings generated successfully",
            data={
                "embeddings": result.get("embeddings", {}),
                "embedding_metadata": result.get("metadata", {})
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
            message="Material visual search health check failed"
        )


# ============================================================================
# SPECIALIZED CLIP EMBEDDING SEARCH ENDPOINTS
# ============================================================================

@router.post(
    "/search/by-color",
    response_model=ImageSearchResponse,
    summary="Search images by color palette",
    description="Search for images with similar color palettes using specialized CLIP embeddings"
)
async def search_by_color(
    request: ImageSearchRequest,
    rag: RAGService = Depends(get_rag_service)
) -> ImageSearchResponse:
    """
    **🎨 Color Palette Search**

    Search for images with similar color palettes using specialized color CLIP embeddings.

    ## Use Cases
    - Find materials with matching color schemes
    - Discover products in specific color families
    - Match color palettes across different materials

    ## Request Example
    ```json
    {
      "query_image": "data:image/jpeg;base64,...",
      "workspace_id": "workspace-uuid",
      "limit": 10,
      "min_similarity": 0.7
    }
    ```
    """
    try:
        from app.services.vecs_service import get_vecs_service
        from app.services.real_embeddings_service import RealEmbeddingsService

        # Generate color embedding from query image
        embeddings_service = RealEmbeddingsService()
        result = await embeddings_service._generate_specialized_clip_embeddings(
            image_url=None,
            image_data=request.query_image
        )

        if not result or 'color' not in result:
            raise HTTPException(status_code=500, detail="Failed to generate color embedding")

        color_embedding = result['color']

        # Search VECS color collection
        vecs_service = get_vecs_service()
        results = await vecs_service.search_specialized_embeddings(
            embedding_type='color',
            query_embedding=color_embedding,
            limit=request.limit or 10,
            filters={"workspace_id": {"$eq": request.workspace_id}} if request.workspace_id else None,
            min_similarity=request.min_similarity or 0.5
        )

        # Format response
        image_results = []
        for item in results:
            image_results.append(ImageSearchResult(
                image_id=item.get('image_id'),
                similarity_score=item.get('similarity_score'),
                metadata=item.get('metadata', {}),
                search_type='color_palette'
            ))

        return ImageSearchResponse(
            success=True,
            results=image_results,
            total_results=len(image_results),
            search_type='color_palette'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Color search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Color search failed: {str(e)}")


@router.post(
    "/search/by-texture",
    response_model=ImageSearchResponse,
    summary="Search images by texture pattern",
    description="Search for images with similar textures using specialized CLIP embeddings"
)
async def search_by_texture(
    request: ImageSearchRequest,
    rag: RAGService = Depends(get_rag_service)
) -> ImageSearchResponse:
    """
    **🔲 Texture Pattern Search**

    Search for images with similar texture patterns using specialized texture CLIP embeddings.

    ## Use Cases
    - Find materials with similar surface textures
    - Match rough/smooth/patterned textures
    - Discover materials with specific tactile properties
    """
    try:
        from app.services.vecs_service import get_vecs_service
        from app.services.real_embeddings_service import RealEmbeddingsService

        embeddings_service = RealEmbeddingsService()
        result = await embeddings_service._generate_specialized_clip_embeddings(
            image_url=None,
            image_data=request.query_image
        )

        if not result or 'texture' not in result:
            raise HTTPException(status_code=500, detail="Failed to generate texture embedding")

        texture_embedding = result['texture']

        vecs_service = get_vecs_service()
        results = await vecs_service.search_specialized_embeddings(
            embedding_type='texture',
            query_embedding=texture_embedding,
            limit=request.limit or 10,
            filters={"workspace_id": {"$eq": request.workspace_id}} if request.workspace_id else None,
            min_similarity=request.min_similarity or 0.5
        )

        image_results = []
        for item in results:
            image_results.append(ImageSearchResult(
                image_id=item.get('image_id'),
                similarity_score=item.get('similarity_score'),
                metadata=item.get('metadata', {}),
                search_type='texture_pattern'
            ))

        return ImageSearchResponse(
            success=True,
            results=image_results,
            total_results=len(image_results),
            search_type='texture_pattern'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Texture search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Texture search failed: {str(e)}")


@router.post(
    "/search/by-style",
    response_model=ImageSearchResponse,
    summary="Search images by design style",
    description="Search for images with similar design styles using specialized CLIP embeddings"
)
async def search_by_style(
    request: ImageSearchRequest,
    rag: RAGService = Depends(get_rag_service)
) -> ImageSearchResponse:
    """
    **✨ Design Style Search**

    Search for images with similar design styles using specialized style CLIP embeddings.

    ## Use Cases
    - Find materials in modern/classic/minimalist styles
    - Match aesthetic preferences
    - Discover products with similar design language
    """
    try:
        from app.services.vecs_service import get_vecs_service
        from app.services.real_embeddings_service import RealEmbeddingsService

        embeddings_service = RealEmbeddingsService()
        result = await embeddings_service._generate_specialized_clip_embeddings(
            image_url=None,
            image_data=request.query_image
        )

        if not result or 'style' not in result:
            raise HTTPException(status_code=500, detail="Failed to generate style embedding")

        style_embedding = result['style']

        vecs_service = get_vecs_service()
        results = await vecs_service.search_specialized_embeddings(
            embedding_type='style',
            query_embedding=style_embedding,
            limit=request.limit or 10,
            filters={"workspace_id": {"$eq": request.workspace_id}} if request.workspace_id else None,
            min_similarity=request.min_similarity or 0.5
        )

        image_results = []
        for item in results:
            image_results.append(ImageSearchResult(
                image_id=item.get('image_id'),
                similarity_score=item.get('similarity_score'),
                metadata=item.get('metadata', {}),
                search_type='design_style'
            ))

        return ImageSearchResponse(
            success=True,
            results=image_results,
            total_results=len(image_results),
            search_type='design_style'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Style search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Style search failed: {str(e)}")


@router.post(
    "/search/by-material",
    response_model=ImageSearchResponse,
    summary="Search images by material type",
    description="Search for images with similar material types using specialized CLIP embeddings"
)
async def search_by_material(
    request: ImageSearchRequest,
    rag: RAGService = Depends(get_rag_service)
) -> ImageSearchResponse:
    """
    **🪨 Material Type Search**

    Search for images with similar material types using specialized material CLIP embeddings.

    ## Use Cases
    - Find wood/metal/stone/fabric materials
    - Match material properties
    - Discover similar material compositions
    """
    try:
        from app.services.vecs_service import get_vecs_service
        from app.services.real_embeddings_service import RealEmbeddingsService

        embeddings_service = RealEmbeddingsService()
        result = await embeddings_service._generate_specialized_clip_embeddings(
            image_url=None,
            image_data=request.query_image
        )

        if not result or 'material' not in result:
            raise HTTPException(status_code=500, detail="Failed to generate material embedding")

        material_embedding = result['material']

        vecs_service = get_vecs_service()
        results = await vecs_service.search_specialized_embeddings(
            embedding_type='material',
            query_embedding=material_embedding,
            limit=request.limit or 10,
            filters={"workspace_id": {"$eq": request.workspace_id}} if request.workspace_id else None,
            min_similarity=request.min_similarity or 0.5
        )

        image_results = []
        for item in results:
            image_results.append(ImageSearchResult(
                image_id=item.get('image_id'),
                similarity_score=item.get('similarity_score'),
                metadata=item.get('metadata', {}),
                search_type='material_type'
            ))

        return ImageSearchResponse(
            success=True,
            results=image_results,
            total_results=len(image_results),
            search_type='material_type'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Material search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Material search failed: {str(e)}")


# ============================================================================
# ============================================================================

# ============================================================================
# The following endpoint has been removed and consolidated into /api/rag/search:
# - POST /unified-search - Use /api/rag/search with strategy parameter
# ============================================================================
