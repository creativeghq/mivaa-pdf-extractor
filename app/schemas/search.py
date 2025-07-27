"""
Search and RAG Pydantic schemas.

This module contains schemas for search functionality, RAG queries,
and semantic similarity operations.
"""

from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator

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
    search_type: str = Field("hybrid", regex="^(semantic|keyword|hybrid)$", description="Type of search")
    
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
    
    # Context information
    page_number: int = Field(..., description="Source page number")
    context_before: Optional[str] = Field(None, description="Text before the match")
    context_after: Optional[str] = Field(None, description="Text after the match")
    
    # Highlighting
    highlighted_content: Optional[str] = Field(None, description="Content with search terms highlighted")
    
    # Metadata
    document_tags: List[str] = Field(default_factory=list, description="Document tags")
    chunk_metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")


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


class QueryRequest(BaseModel):
    """Request model for RAG-based question answering."""
    
    question: str = Field(..., min_length=1, max_length=2000, description="Question to answer")
    context_documents: Optional[List[str]] = Field(None, description="Specific documents to use as context")
    context_tags: Optional[List[str]] = Field(None, description="Filter context by tags")
    
    # RAG parameters
    max_context_chunks: int = Field(5, ge=1, le=20, description="Maximum context chunks to retrieve")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Response creativity (0=focused, 2=creative)")
    max_tokens: int = Field(500, ge=50, le=2000, description="Maximum response length")
    
    # Response options
    include_sources: bool = Field(True, description="Include source citations in response")
    include_confidence: bool = Field(True, description="Include confidence score")
    response_format: str = Field("markdown", regex="^(text|markdown|json)$", description="Response format")
    
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


class SourceCitation(BaseModel):
    """Source citation for RAG responses."""
    
    document_id: str = Field(..., description="Source document ID")
    document_name: str = Field(..., description="Document name")
    chunk_id: str = Field(..., description="Source chunk ID")
    page_number: int = Field(..., description="Page number")
    relevance_score: float = Field(..., description="Relevance to the question")
    excerpt: str = Field(..., description="Relevant text excerpt")


class QueryResponse(BaseResponse):
    """Response model for RAG-based queries."""
    
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    
    # Quality metrics
    confidence_score: Optional[float] = Field(None, description="Answer confidence (0-1)")
    completeness_score: Optional[float] = Field(None, description="Answer completeness (0-1)")
    
    # Source information
    sources: List[SourceCitation] = Field(default_factory=list, description="Source citations")
    context_used: int = Field(..., description="Number of context chunks used")
    
    # Processing metadata
    processing_time_ms: float = Field(..., description="Query processing time")
    model_used: Optional[str] = Field(None, description="AI model used for generation")
    
    class Config:
        schema_extra = {
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
    
    @validator('reference_text')
    def validate_reference(cls, v, values):
        reference_doc = values.get('reference_document_id')
        if not reference_doc and not v:
            raise ValueError('Either reference_document_id or reference_text must be provided')
        if reference_doc and v:
            raise ValueError('Provide either reference_document_id or reference_text, not both')
        return v
    
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