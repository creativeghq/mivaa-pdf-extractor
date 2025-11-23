"""
Pydantic models for RAG API routes

Shared request/response models used across multiple route modules.
"""

from typing import Dict, List, Optional, Any
try:
    from pydantic import BaseModel, Field, field_validator as validator
except ImportError:
    from pydantic import BaseModel, Field, validator


# ============================================================================
# Document Models
# ============================================================================
class DocumentUploadRequest(BaseModel):
    """Request model for document upload and processing."""
    title: Optional[str] = Field(None, description="Document title")
    description: Optional[str] = Field(None, description="Document description")
    tags: Optional[List[str]] = Field(default_factory=list, description="Document tags")
    chunk_size: Optional[int] = Field(1000, ge=100, le=4000, description="Chunk size for processing")
    chunk_overlap: Optional[int] = Field(200, ge=0, le=1000, description="Chunk overlap")
    enable_embedding: bool = Field(True, description="Enable automatic embedding generation")
    processing_mode: Optional[str] = Field("standard", description="Processing mode")
    categories: Optional[str] = Field("all", description="Categories to extract")
    file_url: Optional[str] = Field(None, description="URL to download PDF from")
    discovery_model: Optional[str] = Field("claude", description="AI model for discovery")
    enable_prompt_enhancement: bool = Field(True, description="Enable AI prompt enhancement")
    agent_prompt: Optional[str] = Field(None, description="Natural language instruction")
    workspace_id: Optional[str] = Field("ffafc28b-1b8b-4b0d-b226-9f9a6154004e", description="Workspace ID")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str = Field(..., description="Unique document identifier")
    title: str = Field(..., description="Document title")
    status: str = Field(..., description="Processing status")
    chunks_created: int = Field(..., description="Number of chunks created")
    embeddings_generated: bool = Field(..., description="Whether embeddings were generated")
    processing_time: float = Field(..., description="Processing time in seconds")
    message: str = Field(..., description="Status message")


class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    documents: List[Dict[str, Any]] = Field(..., description="List of documents")
    total_count: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")


# ============================================================================
# Search Models
# ============================================================================
class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., min_length=1, max_length=2000, description="Query text")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of top results")
    similarity_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Similarity threshold")
    include_metadata: bool = Field(True, description="Include document metadata")
    enable_reranking: bool = Field(True, description="Enable result reranking")
    document_ids: Optional[List[str]] = Field(None, description="Filter by document IDs")


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents")
    confidence_score: float = Field(..., description="Confidence score")
    processing_time: float = Field(..., description="Processing time")
    retrieved_chunks: int = Field(..., description="Number of chunks retrieved")


class ChatRequest(BaseModel):
    """Request model for conversational RAG."""
    message: str = Field(..., min_length=1, max_length=2000, description="Chat message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Number of context chunks")
    include_history: bool = Field(True, description="Include conversation history")
    document_ids: Optional[List[str]] = Field(None, description="Filter by document IDs")


class ChatResponse(BaseModel):
    """Response model for conversational RAG."""
    message: str = Field(..., description="Original message")
    response: str = Field(..., description="AI response")
    conversation_id: str = Field(..., description="Conversation ID")
    sources: List[Dict[str, Any]] = Field(..., description="Source documents")
    processing_time: float = Field(..., description="Response generation time")


class SearchRequest(BaseModel):
    """Request model for semantic search."""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    search_type: str = Field("semantic", pattern="^(semantic|hybrid|keyword)$", description="Search type")
    top_k: Optional[int] = Field(10, ge=1, le=50, description="Number of results")
    similarity_threshold: Optional[float] = Field(0.6, ge=0.0, le=1.0, description="Similarity threshold")
    document_ids: Optional[List[str]] = Field(None, description="Filter by document IDs")
    include_content: bool = Field(True, description="Include chunk content")
    workspace_id: str = Field(..., description="Workspace ID")
    include_related_products: bool = Field(True, description="Include related products")
    related_products_limit: int = Field(3, ge=1, le=10, description="Max related products")
    use_search_prompts: bool = Field(True, description="Apply search prompts")
    custom_formatting_prompt: Optional[str] = Field(None, description="Custom formatting prompt")
    material_filters: Optional[Dict[str, Any]] = Field(None, description="Material property filters")
    image_url: Optional[str] = Field(None, description="Image URL for similarity search")
    image_base64: Optional[str] = Field(None, description="Base64-encoded image")


class SearchResponse(BaseModel):
    """Response model for semantic search."""
    query: str = Field(..., description="Original search query")
    enhanced_query: Optional[str] = Field(None, description="Enhanced query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_type: str = Field(..., description="Type of search performed")
    processing_time: float = Field(..., description="Search processing time")
    search_metadata: Optional[Dict[str, Any]] = Field(None, description="Search metadata")


class MMRSearchRequest(BaseModel):
    """Request model for MMR search."""
    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: Optional[int] = Field(10, ge=1, le=50, description="Number of results")
    diversity_threshold: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Diversity threshold")
    lambda_param: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Lambda parameter")
    document_ids: Optional[List[str]] = Field(None, description="Filter by document IDs")
    include_metadata: bool = Field(True, description="Include metadata")


class MMRSearchResponse(BaseModel):
    """Response model for MMR search."""
    query: str = Field(..., description="Original search query")
    results: List[Dict[str, Any]] = Field(..., description="MMR search results")
    total_results: int = Field(..., description="Total number of results")
    diversity_score: float = Field(..., description="Overall diversity score")
    processing_time: float = Field(..., description="Search processing time")


class AdvancedQueryRequest(BaseModel):
    """Request model for advanced query."""
    query: str = Field(..., min_length=1, max_length=2000, description="Query text")
    query_type: str = Field("semantic", pattern="^(factual|analytical|conversational|boolean|fuzzy|semantic)$")
    top_k: Optional[int] = Field(10, ge=1, le=50, description="Number of results")
    enable_expansion: bool = Field(True, description="Enable query expansion")
    enable_rewriting: bool = Field(True, description="Enable query rewriting")
    similarity_threshold: Optional[float] = Field(0.6, ge=0.0, le=1.0)
    document_ids: Optional[List[str]] = Field(None)
    metadata_filters: Optional[Dict[str, Any]] = Field(None)
    search_operator: str = Field("AND", pattern="^(AND|OR|NOT)$")


class AdvancedQueryResponse(BaseModel):
    """Response model for advanced query."""
    original_query: str = Field(..., description="Original query text")
    optimized_query: str = Field(..., description="Optimized/expanded query")
    query_type: str = Field(..., description="Type of query processed")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    expansion_terms: List[str] = Field(..., description="Terms added during expansion")
    processing_time: float = Field(..., description="Query processing time")
    confidence_score: float = Field(..., description="Overall confidence score")


# ============================================================================
# System Models
# ============================================================================
class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    status: str
    llamaindex_available: bool
    message: str

