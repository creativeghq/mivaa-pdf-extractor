"""
Search and RAG Pydantic schemas.

This module contains schemas for search functionality, RAG queries,
and semantic similarity operations.
"""

from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from .common import BaseResponse, PaginationParams
from .documents import DocumentChunk


class SearchRequest(BaseModel):
    """Request model for document search."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    document_ids: Optional[List[str]] = Field(None, description="Limit search to specific documents")
    tags: Optional[List[str]] = Field(None, description="Filter by document tags")
    
    # Search parameters
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score")
    search_type: str = Field("hybrid", pattern="^(semantic|keyword|hybrid|multimodal)$", description="Type of search")
    
    # Multi-modal search parameters
    include_images: bool = Field(False, description="Include image analysis in search results")
    include_ocr_text: bool = Field(True, description="Include OCR-extracted text in search")
    content_types: Optional[List[str]] = Field(None, description="Filter by content types (text, image, mixed)")
    ocr_confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum OCR confidence threshold")
    
    # Filters
    date_from: Optional[str] = Field(None, description="Filter documents from date (ISO format)")
    date_to: Optional[str] = Field(None, description="Filter documents to date (ISO format)")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "machine learning algorithms",
                "document_ids": ["doc_123", "doc_456"],
                "tags": ["research", "ai"],
                "limit": 20,
                "similarity_threshold": 0.75,
                "search_type": "hybrid"
            }
        }


class SearchResult(BaseModel):
    """Individual search result."""
    
    document_id: str = Field(..., description="Source document ID")
    document_name: str = Field(..., description="Document name")
    chunk_id: str = Field(..., description="Matching chunk ID")
    content: str = Field(..., description="Matching content snippet")
    
    # Relevance scoring
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    keyword_score: Optional[float] = Field(None, description="Keyword matching score")
    combined_score: float = Field(..., description="Final combined relevance score")
    
    # Multi-modal scoring
    multimodal_score: Optional[float] = Field(None, description="Multi-modal relevance score (0-1)")
    ocr_confidence: Optional[float] = Field(None, description="OCR extraction confidence (0-1)")
    
    # Context information
    page_number: int = Field(..., description="Source page number")
    context_before: Optional[str] = Field(None, description="Text before the match")
    context_after: Optional[str] = Field(None, description="Text after the match")
    
    # Multi-modal content
    content_type: str = Field("text", description="Content type (text, image, mixed)")
    ocr_text: Optional[str] = Field(None, description="OCR-extracted text content")
    associated_images: List[Dict[str, Any]] = Field(default_factory=list, description="Associated image information")
    image_analysis: Optional[Dict[str, Any]] = Field(None, description="Image analysis results")
    
    # Highlighting
    highlighted_content: Optional[str] = Field(None, description="Content with search terms highlighted")
    
    # Metadata
    document_tags: List[str] = Field(default_factory=list, description="Document tags")
    chunk_metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")

    # Document source information
    filename: Optional[str] = Field(None, description="Original filename of the document")
    processing_status: Optional[str] = Field(None, description="Document processing status")
    created_at: Optional[str] = Field(None, description="Document creation timestamp")
    source_metadata: Dict[str, Any] = Field(default_factory=dict, description="Complete document metadata including file info")


class SearchResponse(BaseResponse):
    """Response model for search operations."""
    
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(..., description="Search results")
    total_found: int = Field(..., description="Total number of matching results")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
    
    # Search metadata
    search_type: str = Field(..., description="Type of search performed")
    filters_applied: Dict[str, Any] = Field(default_factory=dict, description="Applied filters")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "query": "machine learning algorithms",
                "results": [
                    {
                        "document_id": "doc_123",
                        "document_name": "ML Research Paper",
                        "chunk_id": "chunk_456",
                        "content": "Machine learning algorithms are computational methods...",
                        "similarity_score": 0.89,
                        "combined_score": 0.85,
                        "page_number": 3,
                        "highlighted_content": "<mark>Machine learning algorithms</mark> are computational methods...",
                        "document_tags": ["research", "ai"]
                    }
                ],
                "total_found": 15,
                "search_time_ms": 45.2,
                "search_type": "hybrid",
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class DocumentQueryRequest(BaseModel):
    """Request model for document-specific RAG queries."""

    query: str = Field(..., min_length=1, max_length=2000, description="Question or query about the document")
    response_mode: str = Field("compact", description="Response mode (compact, tree_summarize, etc.)")
    include_sources: bool = Field(True, description="Whether to include source citations")
    max_tokens: Optional[int] = Field(None, ge=1, le=4000, description="Maximum tokens in response")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the main findings in this research paper?",
                "response_mode": "compact",
                "include_sources": True,
                "max_tokens": 1000
            }
        }


class DocumentQueryResponse(BaseResponse):
    """Response model for document-specific RAG queries."""

    document_id: str = Field(..., description="ID of the queried document")
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Generated response based on document content")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source citations and references")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "document_id": "doc_123",
                "query": "What are the main findings?",
                "response": "The main findings include...",
                "sources": [
                    {
                        "page": 1,
                        "text": "relevant excerpt",
                        "confidence": 0.95
                    }
                ],
                "metadata": {
                    "document_title": "Research Paper",
                    "response_time": 1.2
                },
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class SemanticSearchRequest(BaseModel):
    """Request model for semantic search across documents."""

    query: str = Field(..., min_length=1, max_length=1000, description="Search query for semantic matching")
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs to search (if None, searches all)")
    similarity_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score threshold")
    max_results: int = Field(20, ge=1, le=100, description="Maximum number of results to return")
    include_metadata: bool = Field(True, description="Whether to include result metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "machine learning algorithms",
                "document_ids": ["doc_123", "doc_456"],
                "similarity_threshold": 0.75,
                "max_results": 10,
                "include_metadata": True
            }
        }


class SemanticSearchResponse(BaseResponse):
    """Response model for semantic search operations."""

    query: str = Field(..., description="Original search query")
    results: List[Dict[str, Any]] = Field(default_factory=list, description="Search results with similarity scores")
    total_results: int = Field(..., description="Total number of results found")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Search metadata and statistics")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "query": "machine learning algorithms",
                "results": [
                    {
                        "document_id": "doc_123",
                        "score": 0.85,
                        "content": "Machine learning algorithms are...",
                        "metadata": {
                            "page": 1,
                            "section": "Introduction"
                        }
                    }
                ],
                "total_results": 5,
                "metadata": {
                    "searched_documents": 10,
                    "similarity_threshold": 0.75,
                    "search_time": 1.2
                },
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class RelatedDocumentsResponse(BaseResponse):
    """Response model for finding related documents."""

    source_document_id: str = Field(..., description="ID of the source document")
    related_documents: List[Dict[str, Any]] = Field(default_factory=list, description="List of related documents with similarity scores")
    total_found: int = Field(..., description="Total number of related documents found")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Search metadata and parameters")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "source_document_id": "doc_123",
                "related_documents": [
                    {
                        "document_id": "doc_456",
                        "title": "Related Research Paper",
                        "similarity_score": 0.85,
                        "content_preview": "This document discusses similar topics...",
                        "metadata": {
                            "author": "Dr. Smith",
                            "date": "2024-07-26"
                        }
                    }
                ],
                "total_found": 3,
                "metadata": {
                    "similarity_threshold": 0.7,
                    "analysis_method": "content_summary_similarity"
                },
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class DocumentSummaryRequest(BaseModel):
    """Request model for document summarization."""

    summary_type: str = Field("brief", description="Type of summary to generate (brief, comprehensive, key_points)")
    max_length: Optional[int] = Field(None, ge=50, le=2000, description="Maximum length of summary in words")
    include_metadata: bool = Field(True, description="Whether to include document metadata in response")

    class Config:
        json_schema_extra = {
            "example": {
                "summary_type": "brief",
                "max_length": 200,
                "include_metadata": True
            }
        }


class DocumentSummaryResponse(BaseResponse):
    """Response model for document summarization."""

    document_id: str = Field(..., description="ID of the summarized document")
    summary_type: str = Field(..., description="Type of summary generated")
    summary: str = Field(..., description="Generated summary text")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Summary metadata and document information")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "document_id": "doc_123",
                "summary_type": "brief",
                "summary": "This document discusses the implementation of AI-powered PDF processing systems...",
                "metadata": {
                    "document_title": "AI PDF Processing Research",
                    "document_length": 5000,
                    "summary_length": 200,
                    "processing_time": 2.5
                },
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class EntityExtractionRequest(BaseModel):
    """Request model for entity extraction from documents."""

    entity_types: List[str] = Field(default=["PERSON", "ORG", "DATE", "LOCATION"], description="Types of entities to extract")
    include_confidence: bool = Field(True, description="Whether to include confidence scores for extracted entities")
    min_confidence: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold for entity extraction")

    class Config:
        json_schema_extra = {
            "example": {
                "entity_types": ["PERSON", "ORG", "DATE", "LOCATION", "MONEY"],
                "include_confidence": True,
                "min_confidence": 0.8
            }
        }


class EntityExtractionResponse(BaseResponse):
    """Response model for entity extraction from documents."""

    document_id: str = Field(..., description="ID of the processed document")
    entity_types: List[str] = Field(..., description="Types of entities that were extracted")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted entities with their details")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extraction metadata and document information")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "document_id": "doc_123",
                "entity_types": ["PERSON", "ORG", "DATE"],
                "entities": [
                    {
                        "text": "John Smith",
                        "type": "PERSON",
                        "confidence": 0.95,
                        "start_pos": 45,
                        "end_pos": 55
                    },
                    {
                        "text": "Microsoft Corporation",
                        "type": "ORG",
                        "confidence": 0.92,
                        "start_pos": 120,
                        "end_pos": 141
                    }
                ],
                "metadata": {
                    "document_title": "Business Report 2024",
                    "extraction_method": "llm_based",
                    "total_entities": 15
                },
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class DocumentComparisonRequest(BaseModel):
    """Request model for comparing multiple documents."""

    document_ids: List[str] = Field(..., min_items=2, max_items=10, description="List of document IDs to compare")
    comparison_aspect: str = Field("content", description="Aspect to compare (content, structure, themes, etc.)")
    include_similarities: bool = Field(True, description="Whether to include similarity analysis")
    include_differences: bool = Field(True, description="Whether to include difference analysis")

    class Config:
        json_schema_extra = {
            "example": {
                "document_ids": ["doc_123", "doc_456", "doc_789"],
                "comparison_aspect": "content",
                "include_similarities": True,
                "include_differences": True
            }
        }


class DocumentComparisonResponse(BaseResponse):
    """Response model for document comparison analysis."""

    document_ids: List[str] = Field(..., description="List of compared document IDs")
    comparison_aspect: str = Field(..., description="Aspect that was compared")
    comparison_result: str = Field(..., description="Detailed comparison analysis")
    documents_info: List[Dict[str, Any]] = Field(default_factory=list, description="Information about compared documents")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Comparison metadata and analysis details")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "document_ids": ["doc_123", "doc_456"],
                "comparison_aspect": "content",
                "comparison_result": "The documents share similar themes around AI and machine learning, but differ in their technical depth...",
                "documents_info": [
                    {
                        "document_id": "doc_123",
                        "title": "AI Research Paper",
                        "content_type": "academic"
                    },
                    {
                        "document_id": "doc_456",
                        "title": "ML Implementation Guide",
                        "content_type": "technical"
                    }
                ],
                "metadata": {
                    "comparison_method": "llm_based_analysis",
                    "similarity_score": 0.75,
                    "analysis_time": 3.2
                },
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class QueryRequest(BaseModel):
    """Request model for RAG-based question answering."""
    
    question: str = Field(..., min_length=1, max_length=2000, description="Question to answer")
    context_documents: Optional[List[str]] = Field(None, description="Specific documents to use as context")
    context_tags: Optional[List[str]] = Field(None, description="Filter context by tags")
    
    # RAG parameters
    max_context_chunks: int = Field(5, ge=1, le=20, description="Maximum context chunks to retrieve")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Response creativity (0=focused, 2=creative)")
    max_tokens: int = Field(500, ge=50, le=2000, description="Maximum response length")
    
    # Multi-modal RAG parameters
    include_image_context: bool = Field(False, description="Include image analysis in context retrieval")
    include_ocr_context: bool = Field(True, description="Include OCR-extracted text in context")
    multimodal_llm_model: Optional[str] = Field(None, description="Specific multi-modal LLM model to use")
    image_analysis_depth: str = Field("standard", pattern="^(basic|standard|detailed)$", description="Level of image analysis")
    
    # Response options
    include_sources: bool = Field(True, description="Include source citations in response")
    include_confidence: bool = Field(True, description="Include confidence score")
    include_image_references: bool = Field(False, description="Include image references in response")
    response_format: str = Field("markdown", pattern="^(text|markdown|json)$", description="Response format")
    
    class Config:
        schema_extra = {
            "example": {
                "question": "What are the main benefits of transformer architectures?",
                "context_documents": ["doc_123", "doc_456"],
                "context_tags": ["ai", "research"],
                "max_context_chunks": 8,
                "temperature": 0.5,
                "include_sources": True,
                "response_format": "markdown"
            }
        }
    
    # Multi-modal specific schemas
    
class ImageSearchRequest(BaseModel):
    """Request model for image-based search with material-specific filtering."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="Image search query")
    document_ids: Optional[List[str]] = Field(None, description="Limit search to specific documents")
    
    # Image search parameters
    limit: int = Field(10, ge=1, le=50, description="Maximum number of results")
    similarity_threshold: float = Field(0.6, ge=0.0, le=1.0, description="Minimum similarity score")
    include_ocr_text: bool = Field(True, description="Include OCR-extracted text in search")
    ocr_confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum OCR confidence threshold")
    
    # Visual similarity parameters
    visual_similarity_threshold: float = Field(0.75, ge=0.0, le=1.0, description="Minimum visual similarity threshold")
    search_type: str = Field("visual_similarity", pattern="^(visual_similarity|semantic_analysis|hybrid|material_properties)$", description="Type of visual search")
    
    # Image analysis parameters
    analysis_depth: str = Field("standard", pattern="^(basic|standard|detailed)$", description="Level of image analysis")
    include_visual_features: bool = Field(True, description="Include visual feature analysis")
    image_analysis_model: Optional[str] = Field(None, description="Specific image analysis model to use")
    
    # Material-specific filtering
    material_filters: Optional[Dict[str, Any]] = Field(None, description="Material property filters")
    material_types: Optional[List[str]] = Field(None, description="Filter by specific material types")
    confidence_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum material analysis confidence")
    
    # Advanced material property filters
    spectral_filters: Optional[Dict[str, Any]] = Field(None, description="Spectral analysis filters")
    chemical_filters: Optional[Dict[str, Any]] = Field(None, description="Chemical composition filters")
    mechanical_filters: Optional[Dict[str, Any]] = Field(None, description="Mechanical property filters")
    
    # Fusion weights for hybrid material search
    fusion_weights: Optional[Dict[str, float]] = Field(None, description="Weights for combining different analysis types")
    
    # Advanced options
    enable_clip_embeddings: bool = Field(True, description="Enable CLIP embedding generation for visual similarity")
    enable_vision_analysis: bool = Field(False, description="Enable vision model analysis for material properties")
    include_analytics: bool = Field(False, description="Include search analytics in response")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "charts and graphs showing revenue data",
                "document_ids": ["doc_123"],
                "limit": 15,
                "similarity_threshold": 0.7,
                "include_ocr_text": True,
                "analysis_depth": "detailed"
            }
        }
    

class ImageSearchResult(BaseModel):
    """Individual image search result."""
    
    document_id: str = Field(..., description="Source document ID")
    document_name: str = Field(..., description="Document name")
    image_id: str = Field(..., description="Image identifier")
    page_number: int = Field(..., description="Source page number")
    
    # Image information
    image_path: Optional[str] = Field(None, description="Path to image file")
    image_dimensions: Optional[Dict[str, int]] = Field(None, description="Image width and height")
    image_format: Optional[str] = Field(None, description="Image format (PNG, JPEG, etc.)")
    
    # Analysis results
    visual_description: Optional[str] = Field(None, description="AI-generated visual description")
    ocr_text: Optional[str] = Field(None, description="OCR-extracted text from image")
    ocr_confidence: Optional[float] = Field(None, description="OCR extraction confidence")
    
    # Relevance scoring
    similarity_score: float = Field(..., description="Visual similarity score (0-1)")
    ocr_relevance_score: Optional[float] = Field(None, description="OCR text relevance score")
    combined_score: float = Field(..., description="Final combined relevance score")
    
    # Visual features
    visual_features: Optional[Dict[str, Any]] = Field(None, description="Extracted visual features")
    detected_objects: List[str] = Field(default_factory=list, description="Detected objects in image")
    
    # Material analysis results
    material_analysis: Optional[Dict[str, Any]] = Field(None, description="Material property analysis results")
    clip_embedding: Optional[List[float]] = Field(None, description="CLIP embedding vector for visual similarity")
    vision_analysis: Optional[Dict[str, Any]] = Field(None, description="Vision model material analysis")
    
    # Material properties
    material_type: Optional[str] = Field(None, description="Identified material type")
    material_confidence: Optional[float] = Field(None, description="Material identification confidence")
    spectral_properties: Optional[Dict[str, Any]] = Field(None, description="Spectral analysis properties")
    chemical_composition: Optional[Dict[str, Any]] = Field(None, description="Chemical composition analysis")
    mechanical_properties: Optional[Dict[str, Any]] = Field(None, description="Mechanical property analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "document_id": "doc_123",
                "document_name": "Financial Report Q3",
                "image_id": "img_456",
                "page_number": 5,
                "visual_description": "Bar chart showing quarterly revenue growth",
                "ocr_text": "Q3 Revenue: $2.5M (+15% YoY)",
                "similarity_score": 0.89,
                "combined_score": 0.85,
                "detected_objects": ["chart", "text", "numbers"]
            }
        }

    
class ImageSearchResponse(BaseResponse):
    """Response model for image search operations."""
    
    query: str = Field(..., description="Original search query")
    results: List[ImageSearchResult] = Field(..., description="Image search results")
    total_found: int = Field(..., description="Total number of matching images")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
    
    # Search metadata
    analysis_depth: str = Field(..., description="Level of analysis performed")
    ocr_enabled: bool = Field(..., description="Whether OCR was enabled")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "query": "charts and graphs",
                "results": [
                    {
                        "document_id": "doc_123",
                        "document_name": "Financial Report",
                        "image_id": "img_456",
                        "page_number": 5,
                        "visual_description": "Bar chart showing revenue data",
                        "similarity_score": 0.89,
                        "combined_score": 0.85
                    }
                ],
                "total_found": 12,
                "search_time_ms": 234.5,
                "analysis_depth": "standard",
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }
    
    
class MultiModalAnalysisRequest(BaseModel):
    """Request model for multi-modal document analysis with material-specific capabilities."""
    
    document_id: str = Field(..., description="Document ID to analyze")
    analysis_types: List[str] = Field(..., description="Types of analysis to perform")
    
    # Analysis parameters
    include_text_analysis: bool = Field(True, description="Include text content analysis")
    include_image_analysis: bool = Field(True, description="Include image content analysis")
    include_ocr_analysis: bool = Field(True, description="Include OCR text analysis")
    include_structure_analysis: bool = Field(False, description="Include document structure analysis")
    
    # OCR parameters
    ocr_language: str = Field("en", description="OCR language code")
    ocr_confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum OCR confidence")
    
    # Image analysis parameters
    image_analysis_depth: str = Field("standard", pattern="^(basic|standard|detailed)$", description="Image analysis depth")
    detect_objects: bool = Field(True, description="Detect objects in images")
    extract_visual_features: bool = Field(True, description="Extract visual features")
    
    # Material-specific analysis parameters
    enable_material_analysis: bool = Field(False, description="Enable material property analysis")
    material_analysis_types: List[str] = Field(default_factory=list, description="Types of material analysis (spectral, chemical, mechanical, thermal)")
    enable_clip_embeddings: bool = Field(False, description="Generate CLIP embeddings for visual similarity")
    enable_vision_analysis: bool = Field(False, description="Use Qwen Vision for material understanding")
    
    # Advanced material analysis
    spectral_analysis: bool = Field(False, description="Perform spectral analysis on materials")
    chemical_analysis: bool = Field(False, description="Perform chemical composition analysis")
    mechanical_analysis: bool = Field(False, description="Perform mechanical property analysis")
    thermal_analysis: bool = Field(False, description="Perform thermal property analysis")
    
    # Multi-modal integration
    multimodal_llm_model: Optional[str] = Field(None, description="Specific multi-modal LLM model to use")
    cross_modal_analysis: bool = Field(False, description="Analyze relationships between different modalities")
    
    # Processing options
    analysis_depth: str = Field("standard", pattern="^(basic|standard|detailed|comprehensive)$", description="Overall analysis depth")
    prioritize_materials: bool = Field(False, description="Prioritize material-related content in analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "document_id": "doc_123",
                "analysis_types": ["text", "image", "ocr"],
                "include_image_analysis": True,
                "image_analysis_depth": "detailed",
                "ocr_language": "en"
            }
        }
    
    
class MultiModalAnalysisResponse(BaseResponse):
    """Response model for multi-modal analysis."""
    
    document_id: str = Field(..., description="Analyzed document ID")
    document_name: str = Field(..., description="Document name")
    
    # Analysis results
    text_analysis: Optional[Dict[str, Any]] = Field(None, description="Text analysis results")
    image_analysis: Optional[Dict[str, Any]] = Field(None, description="Image analysis results")
    ocr_analysis: Optional[Dict[str, Any]] = Field(None, description="OCR analysis results")
    structure_analysis: Optional[Dict[str, Any]] = Field(None, description="Document structure analysis")
    
    # Material-specific analysis results
    material_analysis: Optional[Dict[str, Any]] = Field(None, description="Comprehensive material analysis results")
    spectral_analysis: Optional[Dict[str, Any]] = Field(None, description="Spectral analysis results")
    chemical_analysis: Optional[Dict[str, Any]] = Field(None, description="Chemical composition analysis")
    mechanical_analysis: Optional[Dict[str, Any]] = Field(None, description="Mechanical property analysis")
    thermal_analysis: Optional[Dict[str, Any]] = Field(None, description="Thermal property analysis")
    
    # Visual embeddings and analysis
    clip_embeddings: Optional[List[List[float]]] = Field(None, description="Generated CLIP embeddings for visual similarity")
    vision_analysis: Optional[Dict[str, Any]] = Field(None, description="Qwen Vision material understanding results")
    
    # Combined insights
    multimodal_insights: Optional[Dict[str, Any]] = Field(None, description="Combined multi-modal insights")
    cross_modal_insights: Optional[Dict[str, Any]] = Field(None, description="Cross-modal relationship insights")
    content_summary: Optional[str] = Field(None, description="Overall content summary")
    material_summary: Optional[str] = Field(None, description="Material-focused analysis summary")
    
    # Processing metadata
    analysis_time_ms: float = Field(..., description="Total analysis time")
    models_used: Dict[str, str] = Field(default_factory=dict, description="Models used for analysis")
    material_analysis_enabled: bool = Field(False, description="Whether material analysis was performed")
    
    # Statistics
    total_pages: int = Field(..., description="Total pages analyzed")
    total_images: int = Field(..., description="Total images analyzed")
    total_text_chunks: int = Field(..., description="Total text chunks analyzed")
    total_materials_identified: int = Field(0, description="Total materials identified in analysis")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "document_id": "doc_123",
                "document_name": "Financial Report Q3",
                "text_analysis": {
                    "key_topics": ["revenue", "growth", "market"],
                    "sentiment": "positive"
                },
                "image_analysis": {
                    "chart_count": 5,
                    "table_count": 3,
                    "detected_objects": ["charts", "tables", "logos"]
                },
                "content_summary": "Financial report showing positive Q3 growth with detailed charts and tables",
                "analysis_time_ms": 1250.0,
                "total_pages": 25,
                "total_images": 8,
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }
        


class SourceCitation(BaseModel):
    """Source citation for RAG responses."""
    
    document_id: str = Field(..., description="Source document ID")
    document_name: str = Field(..., description="Document name")
    chunk_id: str = Field(..., description="Source chunk ID")
    page_number: int = Field(..., description="Page number")
    relevance_score: float = Field(..., description="Relevance to the question")
    excerpt: str = Field(..., description="Relevant text excerpt")
    
    # Multi-modal citation fields
    content_type: str = Field("text", description="Content type (text, image, mixed)")
    ocr_excerpt: Optional[str] = Field(None, description="OCR-extracted text excerpt")
    image_reference: Optional[Dict[str, Any]] = Field(None, description="Associated image information")
    multimodal_confidence: Optional[float] = Field(None, description="Multi-modal analysis confidence")


class QueryResponse(BaseResponse):
    """Response model for RAG-based queries."""
    
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    
    # Quality metrics
    confidence_score: Optional[float] = Field(None, description="Answer confidence (0-1)")
    completeness_score: Optional[float] = Field(None, description="Answer completeness (0-1)")
    multimodal_confidence: Optional[float] = Field(None, description="Multi-modal analysis confidence (0-1)")
    
    # Source information
    sources: List[SourceCitation] = Field(default_factory=list, description="Source citations")
    context_used: int = Field(..., description="Number of context chunks used")
    image_context_used: int = Field(0, description="Number of image contexts used")
    
    # Multi-modal response fields
    image_references: List[Dict[str, Any]] = Field(default_factory=list, description="Referenced images in response")
    multimodal_analysis: Optional[Dict[str, Any]] = Field(None, description="Multi-modal analysis results")
    
    # Processing metadata
    processing_time_ms: float = Field(..., description="Query processing time")
    model_used: Optional[str] = Field(None, description="AI model used for generation")
    multimodal_model_used: Optional[str] = Field(None, description="Multi-modal model used for analysis")
    
    class Config:
        model_config = {"protected_namespaces": ()}
        json_schema_extra = {
            "example": {
                "success": True,
                "question": "What are the main benefits of transformer architectures?",
                "answer": "Transformer architectures offer several key benefits:\n\n1. **Parallel Processing**: Unlike RNNs...",
                "confidence_score": 0.92,
                "completeness_score": 0.88,
                "sources": [
                    {
                        "document_id": "doc_123",
                        "document_name": "Attention Is All You Need",
                        "chunk_id": "chunk_456",
                        "page_number": 3,
                        "relevance_score": 0.95,
                        "excerpt": "The Transformer allows for significantly more parallelization..."
                    }
                ],
                "context_used": 5,
                "processing_time_ms": 1250.5,
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }


class SimilaritySearchRequest(BaseModel):
    """Request model for similarity-based document search."""
    
    reference_document_id: Optional[str] = Field(None, description="Find documents similar to this one")
    reference_text: Optional[str] = Field(None, description="Find documents similar to this text")
    
    # Search parameters
    limit: int = Field(10, ge=1, le=50, description="Maximum number of similar documents")
    similarity_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Minimum similarity score")
    exclude_self: bool = Field(True, description="Exclude reference document from results")
    
    # Filters
    tags: Optional[List[str]] = Field(None, description="Filter by document tags")
    document_types: Optional[List[str]] = Field(None, description="Filter by document types")
    
    @model_validator(mode='after')
    def validate_reference(self):
        reference_doc = getattr(self, 'reference_document_id', None)
        reference_text = getattr(self, 'reference_text', None)
        if not reference_doc and not reference_text:
            raise ValueError('Either reference_document_id or reference_text must be provided')
        if reference_doc and reference_text:
            raise ValueError('Provide either reference_document_id or reference_text, not both')
        return self
    
    class Config:
        schema_extra = {
            "example": {
                "reference_document_id": "doc_123",
                "limit": 15,
                "similarity_threshold": 0.7,
                "exclude_self": True,
                "tags": ["research"]
            }
        }


class SimilarDocument(BaseModel):
    """Similar document result."""
    
    document_id: str = Field(..., description="Document ID")
    document_name: str = Field(..., description="Document name")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    
    # Document metadata
    page_count: int = Field(..., description="Number of pages")
    word_count: int = Field(..., description="Word count")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    created_at: str = Field(..., description="Creation timestamp")
    
    # Similarity details
    matching_topics: List[str] = Field(default_factory=list, description="Common topics/themes")
    content_overlap: Optional[float] = Field(None, description="Content overlap percentage")


class SimilaritySearchResponse(BaseResponse):
    """Response model for similarity search."""
    
    reference_info: Dict[str, Any] = Field(..., description="Information about the reference")
    similar_documents: List[SimilarDocument] = Field(..., description="Similar documents found")
    total_found: int = Field(..., description="Total number of similar documents")
    search_time_ms: float = Field(..., description="Search execution time")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "reference_info": {
                    "document_id": "doc_123",
                    "document_name": "AI Research Paper",
                    "type": "document"
                },
                "similar_documents": [
                    {
                        "document_id": "doc_456",
                        "document_name": "Machine Learning Survey",
                        "similarity_score": 0.85,
                        "page_count": 25,
                        "word_count": 8500,
                        "tags": ["ai", "survey"],
                        "created_at": "2024-07-20T10:00:00Z",
                        "matching_topics": ["neural networks", "deep learning"]
                    }
                ],
                "total_found": 8,
                "search_time_ms": 125.3,
                "timestamp": "2024-07-26T18:00:00Z"
            }
        }
