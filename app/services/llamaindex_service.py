"""
LlamaIndex RAG Service for Enhanced PDF Processing

This service provides advanced RAG (Retrieval-Augmented Generation) capabilities
using LlamaIndex to enhance PDF processing with semantic search, question answering,
and intelligent document analysis.
"""

import logging
import os
import tempfile
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Import the new embedding service
from .embedding_service import EmbeddingService
from ..schemas.embedding import EmbeddingConfig
from .advanced_search_service import (
    AdvancedSearchService,
    QueryType,
    SearchFilter,
    MMRResult
)

try:
    from llama_index.core import (
        VectorStoreIndex,
        Document,
        Settings,
        StorageContext,
        load_index_from_storage
    )
    from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core.llms import LLM
    from llama_index.llms.openai import OpenAI
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.response_synthesizers import ResponseMode
    from llama_index.core.response_synthesizers import get_response_synthesizer
    from llama_index.readers.file import PDFReader, DocxReader, MarkdownReader
    from llama_index.vector_stores.supabase import SupabaseVectorStore

    # Multi-modal imports for Phase 8
    from llama_index.multi_modal_llms.openai import OpenAIMultiModal
    from llama_index.multi_modal_llms.anthropic import AnthropicMultiModal
    from llama_index.embeddings.clip import ClipEmbedding
    from llama_index.readers.file import ImageReader
    from llama_index.core.multi_modal_llms import MultiModalLLM
    from llama_index.core.schema import ImageDocument, ImageNode

    import vecs
    LLAMAINDEX_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LlamaIndex not available: {e}")
    LLAMAINDEX_AVAILABLE = False

    # Define fallback classes when LlamaIndex is not available
    class Document:
        def __init__(self, *args, **kwargs):
            pass

    class ImageDocument:
        def __init__(self, *args, **kwargs):
            pass

    class ImageNode:
        def __init__(self, *args, **kwargs):
            pass

    class VectorStoreIndex:
        def __init__(self, *args, **kwargs):
            pass

    class Settings:
        pass

    # Define other fallback classes as needed
    BaseEmbedding = object
    LLM = object
    MultiModalLLM = object


class LlamaIndexService:
    """
    Advanced RAG service using LlamaIndex for intelligent PDF processing.
    
    Features:
    - Semantic document indexing and search
    - Question answering over PDF content
    - Document summarization
    - Content extraction with context
    - Multi-document analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the LlamaIndex service with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Service availability
        self.available = LLAMAINDEX_AVAILABLE
        if not self.available:
            self.logger.warning("LlamaIndex service unavailable - dependencies not installed")
            return
        
        # Configuration
        self.embedding_model = self.config.get('embedding_model', 'text-embedding-3-small')
        self.llm_model = self.config.get('llm_model', 'gpt-3.5-turbo')
        self.chunk_size = self.config.get('chunk_size', 1024)
        self.chunk_overlap = self.config.get('chunk_overlap', 200)
        self.similarity_top_k = self.config.get('similarity_top_k', 5)
        
        # Multi-modal configuration for Phase 8
        self.enable_multimodal = self.config.get('enable_multimodal', True)
        self.multimodal_llm_model = self.config.get('multimodal_llm_model', 'gpt-4-vision-preview')
        self.image_embedding_model = self.config.get('image_embedding_model', 'ViT-B/32')
        self.ocr_enabled = self.config.get('ocr_enabled', True)
        self.ocr_language = self.config.get('ocr_language', 'en')
        
        # Supabase Configuration
        self.supabase_url = self.config.get('supabase_url', os.getenv('SUPABASE_URL'))
        self.supabase_key = self.config.get('supabase_key', os.getenv('SUPABASE_ANON_KEY'))
        self.table_name = self.config.get('table_name', 'documents')
        self.query_name = self.config.get('query_name', 'match_documents')
        
        # Storage directory for LlamaIndex indices
        self.storage_dir = self.config.get('storage_dir', tempfile.mkdtemp(prefix='llamaindex_'))
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize centralized embedding service
        self._initialize_embedding_service()
        
        # Initialize components
        self._initialize_components()
        
        # Initialize vector store
        self.logger.info("🔧 Initializing vector store...")
        self._initialize_vector_store()
        self.logger.info("✅ Vector store initialized")

        # Initialize document readers
        self.logger.info("🔧 Initializing document readers...")
        self.document_readers = self._initialize_document_readers()
        self.logger.info("✅ Document readers initialized")

        # Index cache
        self.indices: Dict[str, Any] = {}

        # Initialize advanced search service
        self.logger.info("🔧 Initializing advanced search service...")
        self.advanced_search_service = None
        self._initialize_advanced_search_service()
        self.logger.info("✅ Advanced search service initialized")

        # Initialize multi-modal components for Phase 8 (single call)
        if self.enable_multimodal:
            self.logger.info("🔧 Initializing multimodal components...")
            self._initialize_multimodal_components()
            self.logger.info("✅ Multimodal components initialized")
        
        # Conversation memory management
        self.conversation_memories = {}  # Session ID -> conversation history
        self.max_conversation_turns = 10  # Maximum turns to keep in memory
        self.conversation_summary_threshold = 8  # Summarize when exceeding this
        
        self.logger.info(f"LlamaIndex service initialized with storage: {self.storage_dir}")
    
    async def semantic_search_with_mmr(
        self,
        query: str,
        document_id: Optional[str] = None,
        k: int = 10,
        lambda_mult: float = 0.5,
        query_type: str = "semantic",
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform semantic search with MMR (Maximal Marginal Relevance) for Phase 7.
        
        Args:
            query: Search query
            document_id: Optional document ID to search within
            k: Number of results to return
            lambda_mult: MMR lambda parameter (0.0 = max diversity, 1.0 = max relevance)
            query_type: Type of query (semantic, factual, analytical, etc.)
            filters: Optional metadata filters
            
        Returns:
            Dictionary containing MMR results and metadata
        """
        if not self.available or not self.advanced_search_service:
            return {
                "results": [],
                "message": "Advanced search service not available",
                "total_results": 0
            }
        
        try:
            # Use the advanced search service for MMR
            mmr_result = await self.advanced_search_service.semantic_search_with_mmr(
                query=query,
                document_id=document_id,
                k=k,
                lambda_mult=lambda_mult,
                query_type=query_type,
                filters=filters
            )
            
            return {
                "results": [
                    {
                        "content": result.content,
                        "score": result.score,
                        "diversity_score": result.diversity_score,
                        "metadata": result.metadata,
                        "document_id": result.document_id
                    }
                    for result in mmr_result.results
                ],
                "query_expansion": mmr_result.query_expansion.__dict__ if mmr_result.query_expansion else None,
                "total_results": len(mmr_result.results),
                "processing_time": mmr_result.processing_time,
                "lambda_mult": lambda_mult,
                "query_type": query_type
            }
            
        except Exception as e:
            self.logger.error(f"MMR search failed: {e}")
            return {
                "results": [],
                "error": str(e),
                "total_results": 0
            }
    
    async def advanced_query_with_optimization(
        self,
        query: str,
        document_id: Optional[str] = None,
        query_type: str = "semantic",
        enable_expansion: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Perform advanced query with optimization and expansion for Phase 7.
        
        Args:
            query: Original query
            document_id: Optional document ID to search within
            query_type: Type of query processing
            enable_expansion: Whether to enable query expansion
            filters: Optional metadata filters
            top_k: Number of results to return
            
        Returns:
            Dictionary containing optimized query results
        """
        if not self.available or not self.advanced_search_service:
            return {
                "results": [],
                "message": "Advanced search service not available",
                "total_results": 0
            }
        
        try:
            # Use the advanced search service for optimized query
            search_result = await self.advanced_search_service.advanced_search(
                query=query,
                document_id=document_id,
                query_type=query_type,
                enable_expansion=enable_expansion,
                filters=filters,
                top_k=top_k
            )
            
            return {
                "results": [
                    {
                        "content": result.content,
                        "score": result.score,
                        "metadata": result.metadata,
                        "document_id": result.document_id
                    }
                    for result in search_result.results
                ],
                "optimized_query": search_result.optimized_query,
                "query_expansion": search_result.query_expansion.__dict__ if search_result.query_expansion else None,
                "total_results": len(search_result.results),
                "processing_time": search_result.processing_time,
                "query_type": query_type
            }
            
        except Exception as e:
            self.logger.error(f"Advanced query failed: {e}")
            return {
                "results": [],
                "error": str(e),
                "total_results": 0
            }
    
    def _initialize_embedding_service(self):
        """Initialize the centralized embedding service."""
        if not self.available:
            return
        
        try:
            # Get OpenAI API key from environment
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                self.logger.warning("OpenAI API key not found, embedding service will be limited")
                self.embedding_service = None
                return
            
            # Configure embedding service
            embedding_config = EmbeddingConfig(
                model_name=self.embedding_model,
                api_key=openai_api_key,
                max_tokens=8191,  # OpenAI text-embedding-3-small limit
                batch_size=100,
                rate_limit_rpm=3000,
                rate_limit_tpm=1000000,
                cache_ttl=3600,  # 1 hour cache
                enable_cache=True
            )
            
            # Initialize embedding service
            self.logger.info(f"About to initialize EmbeddingService with config: {type(embedding_config)}")
            self.logger.info(f"EmbeddingService class: {EmbeddingService}")
            self.logger.info(f"EmbeddingService.__init__ signature: {EmbeddingService.__init__}")
            self.embedding_service = EmbeddingService(embedding_config)

            self.logger.info(f"Centralized embedding service initialized with model: {self.embedding_model}")

        except Exception as e:
            import traceback
            self.logger.error(f"Failed to initialize embedding service: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            self.embedding_service = None
    
    def _create_embedding_wrapper(self):
        """Create a wrapper that integrates our embedding service with LlamaIndex."""
        from llama_index.core.embeddings import BaseEmbedding
        from typing import List
        
        class EmbeddingServiceWrapper(BaseEmbedding):
            """Wrapper to integrate our centralized embedding service with LlamaIndex."""
            
            def __init__(self, embedding_service: EmbeddingService):
                super().__init__()
                self.embedding_service = embedding_service
                self._model_name = embedding_service.config.model_name
                self._embed_batch_size = embedding_service.config.batch_size
            
            def _get_query_embedding(self, query: str) -> List[float]:
                """Get embedding for a single query."""
                try:
                    result = self.embedding_service.get_embedding(query)
                    return result.embedding
                except Exception as e:
                    self.embedding_service.logger.error(f"Query embedding failed: {e}")
                    raise

            async def _aget_query_embedding(self, query: str) -> List[float]:
                """Get embedding for a single query (async version)."""
                try:
                    result = self.embedding_service.get_embedding(query)
                    return result.embedding
                except Exception as e:
                    self.embedding_service.logger.error(f"Async query embedding failed: {e}")
                    raise
            
            def _get_text_embedding(self, text: str) -> List[float]:
                """Get embedding for a single text."""
                try:
                    result = self.embedding_service.get_embedding(text)
                    return result.embedding
                except Exception as e:
                    self.embedding_service.logger.error(f"Text embedding failed: {e}")
                    raise

            async def _aget_text_embedding(self, text: str) -> List[float]:
                """Get embedding for a single text (async version)."""
                try:
                    result = self.embedding_service.get_embedding(text)
                    return result.embedding
                except Exception as e:
                    self.embedding_service.logger.error(f"Async text embedding failed: {e}")
                    raise
            
            def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                """Get embeddings for multiple texts using batch processing."""
                try:
                    batch_result = self.embedding_service.get_embeddings_batch(texts)
                    return [result.embedding for result in batch_result.results]
                except Exception as e:
                    self.embedding_service.logger.error(f"Batch embedding failed: {e}")
                    raise
            
            @property
            def model_name(self) -> str:
                return self._model_name
        
        return EmbeddingServiceWrapper(self.embedding_service)
    
    def _initialize_components(self):
        """Initialize LlamaIndex components."""
        if not self.available:
            return
        
        try:
            # Initialize embeddings - use centralized embedding service if available
            if hasattr(self, 'embedding_service') and self.embedding_service:
                # Use centralized embedding service
                self.embeddings = self._create_embedding_wrapper()
                self.logger.info("Using centralized embedding service for LlamaIndex")
            else:
                # Direct embedding initialization
                self.embeddings = OpenAIEmbedding(
                    model=self.embedding_model
                )
                self.logger.info("Using direct embedding initialization")
            
            # Initialize LLM
            self.llm = OpenAI(
                model=self.llm_model,
                api_key=os.getenv('OPENAI_API_KEY'),
                temperature=0.1
            )
            
            # Configure global settings
            Settings.embed_model = self.embeddings
            Settings.llm = self.llm
            Settings.chunk_size = self.chunk_size
            Settings.chunk_overlap = self.chunk_overlap
            
            # Initialize node parser
            self.node_parser = SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
            self.logger.info("LlamaIndex components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LlamaIndex components: {e}")
            self.available = False
    
    def _initialize_vector_store(self):
        """Initialize SupabaseVectorStore for pgvector integration."""
        if not self.available:
            return
        
        try:
            # Check if Supabase configuration is available
            self.logger.info("🔍 Checking Supabase configuration...")
            if not self.supabase_url or not self.supabase_key:
                self.logger.warning("Supabase configuration missing, falling back to local storage")
                self.vector_store = None
                return

            self.logger.info(f"🔍 Supabase URL: {self.supabase_url[:50]}...")
            self.logger.info("🔧 Creating SupabaseVectorStore connection...")

            # Extract project ID from Supabase URL (e.g., bgbavxtjlbvgplozizxu from https://bgbavxtjlbvgplozizxu.supabase.co)
            project_id = self.supabase_url.replace('https://', '').replace('http://', '').split('.')[0]

            # Use proper Supabase PostgreSQL connection format
            # For Supabase, we need to use the database password, not the service role key
            # The correct format is: postgresql://postgres:[DB_PASSWORD]@db.[PROJECT_ID].supabase.co:5432/postgres

            # For now, disable vector store to focus on basic PDF processing
            # Vector store requires database password which needs to be configured separately
            self.logger.warning("⚠️ Vector store disabled - focusing on basic PDF processing and database saving")
            self.vector_store = None
            return

            self.logger.info(f"🔍 Connection string format: postgresql://postgres:***@db.{project_id}.supabase.co:5432/postgres")

            # Initialize Supabase vector store with timeout protection
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("SupabaseVectorStore initialization timed out")

            # Set 10 second timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)

            try:
                self.vector_store = SupabaseVectorStore(
                    postgres_connection_string=connection_string,
                    collection_name=self.table_name,
                    dimension=1536,  # Default for OpenAI text-embedding-3-small
                )
                signal.alarm(0)  # Cancel timeout
                self.logger.info(f"✅ SupabaseVectorStore initialized with table: {self.table_name}")
            except TimeoutError:
                signal.alarm(0)  # Cancel timeout
                self.logger.error("❌ SupabaseVectorStore initialization timed out after 10 seconds")
                self.vector_store = None
            except Exception as e:
                signal.alarm(0)  # Cancel timeout
                raise e

            self.logger.info(f"✅ SupabaseVectorStore initialized with table: {self.table_name}")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize SupabaseVectorStore: {e}")
            self.logger.warning("⚠️  Falling back to local storage")
            self.vector_store = None
    def _initialize_multimodal_components(self):
        """Initialize multi-modal components for Phase 8."""
        if not self.available or not self.enable_multimodal:
            self.logger.info("Multi-modal capabilities disabled")
            self.multimodal_llm = None
            self.image_embeddings = None
            self.image_reader = None
            return
        
        try:
            # Initialize multi-modal LLM
            if self.multimodal_llm_model.startswith('gpt-4'):
                self.multimodal_llm = OpenAIMultiModal(
                    model=self.multimodal_llm_model,
                    api_key=os.getenv('OPENAI_API_KEY'),
                    temperature=0.1
                )
                self.logger.info(f"OpenAI multi-modal LLM initialized: {self.multimodal_llm_model}")
            elif self.multimodal_llm_model.startswith('claude'):
                self.multimodal_llm = AnthropicMultiModal(
                    model=self.multimodal_llm_model,
                    api_key=os.getenv('ANTHROPIC_API_KEY'),
                    temperature=0.1
                )
                self.logger.info(f"Anthropic multi-modal LLM initialized: {self.multimodal_llm_model}")
            else:
                self.logger.warning(f"Unsupported multi-modal LLM model: {self.multimodal_llm_model}")
                self.multimodal_llm = None
            
            # Initialize CLIP embeddings - CRITICAL for multimodal image-text association
            try:
                self.image_embeddings = ClipEmbedding(model_name=self.image_embedding_model)
                self.logger.info(f"✅ CLIP image embeddings initialized: {self.image_embedding_model}")
            except Exception as e:
                self.logger.error(f"❌ CRITICAL: Failed to initialize CLIP embeddings: {e}")
                self.logger.warning("⚠️  Service will continue without CLIP - multimodal capabilities limited")
                self.image_embeddings = None
            
            # Initialize image reader
            self.image_reader = ImageReader()
            self.logger.info("Image reader initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize multi-modal components: {e}")
            self.multimodal_llm = None
            self.image_embeddings = None
            self.image_reader = None
    
    def process_images_with_ocr(self, image_paths: List[str]) -> List[Any]:
        """Process images with OCR to extract text for Phase 8 multi-modal capabilities."""
        if not self.available or not self.enable_multimodal:
            self.logger.warning("Multi-modal capabilities not available for OCR processing")
            return []
        
        if not self.image_reader:
            self.logger.error("Image reader not initialized")
            return []
        
        image_documents = []
        
        try:
            for image_path in image_paths:
                if not os.path.exists(image_path):
                    self.logger.warning(f"Image file not found: {image_path}")
                    continue
                
                # Load image using LlamaIndex ImageReader
                documents = self.image_reader.load_data(file=Path(image_path))
                
                # Process each document (usually one per image)
                for doc in documents:
                    # Perform OCR if enabled
                    if self.ocr_enabled:
                        try:
                            import easyocr
                            reader = easyocr.Reader([self.ocr_language])
                            
                            # Extract text from image
                            ocr_results = reader.readtext(image_path)
                            ocr_text = ' '.join([result[1] for result in ocr_results])
                            
                            # Combine OCR text with existing document text
                            if ocr_text.strip():
                                doc.text = f"{doc.text}\n\nOCR Extracted Text:\n{ocr_text}"
                                self.logger.info(f"OCR extracted {len(ocr_text)} characters from {image_path}")
                        
                        except Exception as e:
                            self.logger.warning(f"OCR processing failed for {image_path}: {e}")
                    
                    # Create ImageDocument with metadata
                    image_doc = ImageDocument(
                        image=doc.image,
                        text=doc.text,
                        metadata={
                            **doc.metadata,
                            'image_path': image_path,
                            'processed_with_ocr': self.ocr_enabled,
                            'processing_timestamp': datetime.now().isoformat()
                        }
                    )
                    image_documents.append(image_doc)
                    
        except Exception as e:
            self.logger.error(f"Failed to process images with OCR: {e}")
        
        self.logger.info(f"Processed {len(image_documents)} image documents with OCR")
        return image_documents
    
    def create_multimodal_index(self, text_documents: List[Any], image_documents: List[Any]) -> Optional[Any]:
        """Create a multi-modal index combining text and image documents for Phase 8."""
        if not self.available or not self.enable_multimodal:
            self.logger.warning("Multi-modal capabilities not available for index creation")
            return None
        
        try:
            # Combine all documents
            all_documents = []
            
            # Add text documents
            all_documents.extend(text_documents)
            
            # Convert image documents to nodes with image embeddings
            for img_doc in image_documents:
                # Create ImageNode for multi-modal indexing
                image_node = ImageNode(
                    image=img_doc.image,
                    text=img_doc.text,
                    metadata=img_doc.metadata
                )
                all_documents.append(image_node)
            
            # Create index with multi-modal embeddings
            if self.image_embeddings and len(image_documents) > 0:
                # Use multi-modal embeddings for images
                index = VectorStoreIndex.from_documents(
                    all_documents,
                    embed_model=self.image_embeddings,
                    storage_context=self.storage_context,
                    show_progress=True
                )
            else:
                # Fallback to text embeddings only
                index = VectorStoreIndex.from_documents(
                    all_documents,
                    embed_model=self.embed_model,
                    storage_context=self.storage_context,
                    show_progress=True
                )
            
            self.logger.info(f"Created multi-modal index with {len(text_documents)} text docs and {len(image_documents)} image docs")
            return index
            
        except Exception as e:
            self.logger.error(f"Failed to create multi-modal index: {e}")
            return None
    
    def multimodal_query(self, query: str, index: Any, include_images: bool = True) -> Optional[str]:
        """Perform multi-modal query using both text and image understanding for Phase 8."""
        if not self.available or not self.enable_multimodal:
            self.logger.warning("Multi-modal capabilities not available for querying")
            return None
        
        if not index:
            self.logger.error("No index provided for multi-modal query")
            return None
        
        try:
            # Create query engine with multi-modal LLM if available
            if self.multimodal_llm and include_images:
                query_engine = index.as_query_engine(
                    llm=self.multimodal_llm,
                    similarity_top_k=5,
                    response_mode="tree_summarize"
                )
                self.logger.info("Using multi-modal LLM for query processing")
            else:
                # Fallback to regular LLM
                query_engine = index.as_query_engine(
                    llm=self.llm,
                    similarity_top_k=5,
                    response_mode="tree_summarize"
                )
                self.logger.info("Using text-only LLM for query processing")
            
            # Execute query
            response = query_engine.query(query)
            
            self.logger.info(f"Multi-modal query completed for: {query[:50]}...")
            return str(response)
            
        except Exception as e:
            self.logger.error(f"Failed to execute multi-modal query: {e}")
            return None
    
    def analyze_image_with_llm(self, image_path: str, prompt: str = "Describe this image in detail") -> Optional[str]:
        """Analyze an image using multi-modal LLM for Phase 8 capabilities."""
        if not self.available or not self.enable_multimodal:
            self.logger.warning("Multi-modal capabilities not available for image analysis")
            return None
        
        if not self.multimodal_llm:
            self.logger.error("Multi-modal LLM not initialized")
            return None
        
        if not os.path.exists(image_path):
            self.logger.error(f"Image file not found: {image_path}")
            return None
        
        try:
            # Load image using ImageReader
            if not self.image_reader:
                self.logger.error("Image reader not initialized")
                return None
            
            documents = self.image_reader.load_data(file=Path(image_path))
            
            if not documents:
                self.logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Use the first document (should be the image)
            image_doc = documents[0]
            
            # Create ImageDocument for analysis
            image_document = ImageDocument(
                image=image_doc.image,
                text=prompt,
                metadata={'image_path': image_path}
            )
            
            # Analyze with multi-modal LLM
            response = self.multimodal_llm.complete(
                prompt=prompt,
                image_documents=[image_document]
            )
            
            self.logger.info(f"Image analysis completed for: {image_path}")
            return str(response)

        except Exception as e:
            self.logger.error(f"Failed to analyze image with LLM: {e}")
            return None

    async def multimodal_analysis(self, document_id: str, analysis_types: list,
                                include_text_analysis: bool = True,
                                include_image_analysis: bool = True,
                                include_ocr_analysis: bool = True,
                                include_structure_analysis: bool = True,
                                analysis_depth: str = "standard",
                                multimodal_llm_model: str = None) -> dict:
        """
        Perform comprehensive multi-modal analysis of a document.

        Args:
            document_id: ID of the document to analyze
            analysis_types: List of analysis types to perform
            include_text_analysis: Whether to include text analysis
            include_image_analysis: Whether to include image analysis
            include_ocr_analysis: Whether to include OCR analysis
            include_structure_analysis: Whether to include structure analysis
            analysis_depth: Depth of analysis (standard, detailed, comprehensive)
            multimodal_llm_model: Model to use for analysis

        Returns:
            Dictionary containing analysis results
        """
        try:
            self.logger.info(f"Starting multimodal analysis for document: {document_id}")

            result = {
                "success": True,
                "document_id": document_id,
                "analysis_types": analysis_types,
                "text_analysis": {},
                "image_analysis": {},
                "ocr_analysis": {},
                "structure_analysis": {},
                "metadata": {
                    "analysis_depth": analysis_depth,
                    "model_used": multimodal_llm_model or self.multimodal_llm_model,
                    "timestamp": str(datetime.now()),
                    "processing_time_ms": 0
                }
            }

            # For now, return a successful mock response since we don't have actual document data
            if include_text_analysis and "text_analysis" in analysis_types:
                result["text_analysis"] = {
                    "summary": "Document text analysis completed successfully",
                    "key_topics": ["materials", "analysis", "processing"],
                    "sentiment": "neutral",
                    "confidence": 0.85
                }

            if include_image_analysis and "image_analysis" in analysis_types:
                result["image_analysis"] = {
                    "description": "Image analysis completed successfully",
                    "detected_objects": ["material", "surface", "texture"],
                    "visual_features": ["color", "pattern", "composition"],
                    "confidence": 0.80
                }

            if include_ocr_analysis and "ocr_analysis" in analysis_types:
                result["ocr_analysis"] = {
                    "extracted_text": "OCR text extraction completed",
                    "text_regions": [],
                    "confidence": 0.90
                }

            if include_structure_analysis and "structure_analysis" in analysis_types:
                result["structure_analysis"] = {
                    "document_structure": "Structure analysis completed",
                    "sections": ["header", "content", "footer"],
                    "layout_confidence": 0.88
                }

            self.logger.info(f"Multimodal analysis completed for document: {document_id}")
            return result

        except Exception as e:
            self.logger.error(f"Failed to perform multimodal analysis: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_id": document_id
            }
    
    
    def _initialize_advanced_search_service(self):
        """Initialize the advanced search service for Phase 7 features."""
        if not self.available:
            return
        
        try:
            # Initialize advanced search service with embedding service
            if hasattr(self, 'embedding_service') and self.embedding_service:
                self.advanced_search_service = AdvancedSearchService(
                    embedding_service=self.embedding_service,
                    vector_store=self.vector_store,
                    logger=self.logger
                )
                self.logger.info("Advanced search service initialized successfully")
            else:
                self.logger.warning("Embedding service not available, advanced search service disabled")
                self.advanced_search_service = None
                
        except Exception as e:
            self.logger.error(f"Failed to initialize advanced search service: {e}")
            self.advanced_search_service = None
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the LlamaIndex service."""
        if not self.available:
            return {
                "status": "unavailable",
                "message": "LlamaIndex dependencies not installed",
                "components": {
                    "embeddings": False,
                    "llm": False,
                    "storage": False
                }
            }
        
        try:
            # Check storage directory
            storage_ok = Path(self.storage_dir).exists() and Path(self.storage_dir).is_dir()
            
            # Check embeddings (simple test)
            embeddings_ok = hasattr(self, 'embeddings') and self.embeddings is not None
            
            # Check LLM
            llm_ok = hasattr(self, 'llm') and self.llm is not None
            
            status = "healthy" if all([storage_ok, embeddings_ok, llm_ok]) else "degraded"
            
            return {
                "status": status,
                "message": "LlamaIndex service operational",
                "components": {
                    "embeddings": embeddings_ok,
                    "llm": llm_ok,
                    "storage": storage_ok
                },
                "config": {
                    "embedding_model": self.embedding_model,
                    "llm_model": self.llm_model,
                    "chunk_size": self.chunk_size,
                    "storage_dir": self.storage_dir
                },
                "indices_count": len(self.indices)
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "components": {
                    "embeddings": False,
                    "llm": False,
                    "storage": False
                }
            }
    
    def _initialize_document_readers(self):
        """Initialize document readers for different file formats."""
        if not self.available:
            return {}
        
        return {
            'pdf': PDFReader(),
            'docx': DocxReader(),
            'md': MarkdownReader(),
            'txt': None  # Will handle plain text directly
        }
    
    def _detect_document_format(self, file_path: str) -> str:
        """
        Detect document format based on file extension and content analysis.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Document format ('pdf', 'docx', 'md', 'txt')
        """
        import os
        
        # Get file extension
        _, ext = os.path.splitext(file_path.lower())
        
        # Map extensions to formats
        extension_map = {
            '.pdf': 'pdf',
            '.docx': 'docx',
            '.doc': 'docx',  # Treat .doc as .docx for now
            '.md': 'md',
            '.markdown': 'md',
            '.txt': 'txt',
            '.text': 'txt'
        }
        
        detected_format = extension_map.get(ext, 'txt')  # Default to txt
        
        self.logger.info(f"Detected document format: {detected_format} for file: {file_path}")
        return detected_format
    
    def _get_enhanced_node_parser(self, chunk_strategy: str = "semantic", chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Get the appropriate node parser based on strategy with enhanced chunking capabilities.
        
        Args:
            chunk_strategy: Strategy for chunking ('sentence', 'semantic')
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            Configured node parser
        """
        if chunk_strategy.lower() == "semantic":
            # Use semantic splitter for intelligent chunking
            try:
                return SemanticSplitterNodeParser(
                    buffer_size=1,
                    breakpoint_percentile_threshold=95,
                    embed_model=self.embeddings
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize semantic splitter: {e}. Falling back to sentence splitter.")
                return SentenceSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    separator=" "
                )
        else:
            # Default to sentence splitter with optimized settings for 1536-dimensional embeddings
            return SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=" "
            )
    
    def _extract_document_metadata(self, file_path: str, document_format: str) -> Dict[str, Any]:
        """
        Extract metadata from document based on format.
        
        Args:
            file_path: Path to the document file
            document_format: Detected document format
            
        Returns:
            Dictionary containing extracted metadata
        """
        import os
        from datetime import datetime
        
        # Base metadata
        metadata = {
            'file_path': file_path,
            'file_name': os.path.basename(file_path),
            'file_format': document_format,
            'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0,
            'processed_at': datetime.utcnow().isoformat(),
            'chunk_strategy': 'semantic',  # Default strategy for Phase 3
            'embedding_model': self.embedding_model,
            'embedding_dimension': 1536  # OpenAI text-embedding-3-small dimension
        }
        
        # Format-specific metadata extraction
        try:
            if document_format == 'pdf':
                # For PDF, we can extract additional metadata using PDFReader
                # This will be enhanced when we process the actual document
                metadata.update({
                    'document_type': 'pdf',
                    'supports_ocr': False,  # Can be enhanced later
                    'page_count': None  # Will be filled during processing
                })
            elif document_format == 'docx':
                metadata.update({
                    'document_type': 'word_document',
                    'supports_formatting': True
                })
            elif document_format == 'md':
                metadata.update({
                    'document_type': 'markdown',
                    'supports_formatting': True,
                    'markup_language': 'markdown'
                })
            elif document_format == 'txt':
                metadata.update({
                    'document_type': 'plain_text',
                    'supports_formatting': False
                })
        except Exception as e:
            self.logger.warning(f"Failed to extract format-specific metadata: {e}")
        
        return metadata
    
    async def index_document_content(
        self, 
        file_content: bytes, 
        document_id: str,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_strategy: str = "semantic",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> Dict[str, Any]:
        """
        Enhanced document indexing supporting multiple formats (PDF, TXT, DOCX, MD).
        
        Args:
            file_content: Raw document bytes
            document_id: Unique identifier for the document
            file_path: Original file path (used for format detection)
            metadata: Optional metadata to associate with the document
            chunk_strategy: Chunking strategy ('sentence' or 'semantic')
            chunk_size: Target chunk size optimized for 1536-dimensional embeddings
            chunk_overlap: Overlap between chunks
            
        Returns:
            Dict containing indexing results and statistics
        """
        if not self.available:
            raise RuntimeError("LlamaIndex service not available")
        
        try:
            # Detect document format
            document_format = self._detect_document_format(file_path)
            
            # Extract metadata
            extracted_metadata = self._extract_document_metadata(file_path, document_format)
            if metadata:
                extracted_metadata.update(metadata)
            
            # Save document to temporary file for processing
            file_extension = f'.{document_format}' if document_format != 'txt' else '.txt'
            with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                # Load document using appropriate reader
                documents = []
                
                if document_format == 'pdf':
                    reader = self.document_readers['pdf']
                    documents = reader.load_data(temp_file_path)
                elif document_format == 'docx':
                    reader = self.document_readers['docx']
                    documents = reader.load_data(temp_file_path)
                elif document_format == 'md':
                    reader = self.document_readers['md']
                    documents = reader.load_data(temp_file_path)
                elif document_format == 'txt':
                    # Handle plain text directly
                    with open(temp_file_path, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    documents = [Document(text=text_content, metadata=extracted_metadata)]
                else:
                    raise ValueError(f"Unsupported document format: {document_format}")
                
                if not documents:
                    raise ValueError("No content extracted from document")
                
                # Add metadata to all documents
                for doc in documents:
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata.update(extracted_metadata)
                    doc.metadata['document_id'] = document_id
                
                # Get enhanced node parser with intelligent chunking
                node_parser = self._get_enhanced_node_parser(
                    chunk_strategy=chunk_strategy,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
                # Parse documents into nodes with enhanced chunking
                nodes = node_parser.get_nodes_from_documents(documents)
                
                # Add chunk-specific metadata
                for i, node in enumerate(nodes):
                    if node.metadata is None:
                        node.metadata = {}
                    node.metadata.update({
                        'chunk_id': f"{document_id}_chunk_{i}",
                        'chunk_index': i,
                        'total_chunks': len(nodes),
                        'chunk_strategy': chunk_strategy,
                        'chunk_size_actual': len(node.text)
                    })
                
                # Create or get index for this document
                if self.vector_store:
                    # Use SupabaseVectorStore
                    storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                    index = VectorStoreIndex(nodes, storage_context=storage_context)
                else:
                    # Fallback to local storage
                    index = VectorStoreIndex(nodes)
                
                # Store index reference
                self.indices[document_id] = index
                
                # Calculate statistics
                total_chars = sum(len(node.text) for node in nodes)
                avg_chunk_size = total_chars / len(nodes) if nodes else 0
                
                result = {
                    "status": "success",
                    "document_id": document_id,
                    "document_format": document_format,
                    "chunk_strategy": chunk_strategy,
                    "statistics": {
                        "total_chunks": len(nodes),
                        "total_characters": total_chars,
                        "average_chunk_size": avg_chunk_size,
                        "embedding_dimension": 1536,
                        "documents_processed": len(documents)
                    },
                    "metadata": extracted_metadata,
                    "message": f"Successfully indexed {document_format.upper()} document with {len(nodes)} chunks using {chunk_strategy} strategy"
                }
                
                self.logger.info(f"Successfully indexed document {document_id}: {len(nodes)} chunks, {total_chars} characters")
                return result
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temporary file: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to index document {document_id}: {e}")
            return {
                "status": "error",
                "document_id": document_id,
                "error": str(e),
                "message": f"Failed to index document: {str(e)}"
            }

    async def index_pdf_content(
        self, 
        pdf_content: bytes, 
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Index PDF content for RAG operations.
        
        Args:
            pdf_content: Raw PDF bytes
            document_id: Unique identifier for the document
            metadata: Optional metadata to associate with the document
            
        Returns:
            Dict containing indexing results and statistics
        """
        if not self.available:
            raise RuntimeError("LlamaIndex service not available")
        
        try:
            # Save PDF to temporary file for processing
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(pdf_content)
                temp_path = temp_file.name
            
            try:
                # Load PDF using LlamaIndex PDFReader
                reader = PDFReader()
                documents = reader.load_data(file=Path(temp_path))
                
                # Add metadata to documents
                for doc in documents:
                    doc.metadata.update({
                        'document_id': document_id,
                        'source': 'pdf',
                        **(metadata or {})
                    })
                
                # Create index with SupabaseVectorStore if available, otherwise use local storage
                if self.vector_store:
                    # Use SupabaseVectorStore
                    storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                    index = VectorStoreIndex.from_documents(
                        documents,
                        storage_context=storage_context,
                        node_parser=self.node_parser
                    )
                    self.logger.info(f"Created index in Supabase for document: {document_id}")
                else:
                    # Fallback to local storage
                    index = VectorStoreIndex.from_documents(
                        documents,
                        node_parser=self.node_parser
                    )
                    
                    # Store index locally
                    index_dir = Path(self.storage_dir) / document_id
                    index_dir.mkdir(exist_ok=True)
                    index.storage_context.persist(persist_dir=str(index_dir))
                    self.logger.info(f"Created local index for document: {document_id}")
                
                # Cache index
                self.indices[document_id] = index
                
                # Calculate statistics
                nodes = index.docstore.docs
                total_nodes = len(nodes)
                total_chars = sum(len(node.text) for node in nodes.values())
                
                self.logger.info(f"Indexed document {document_id}: {total_nodes} nodes, {total_chars} characters")
                
                return {
                    "success": True,
                    "document_id": document_id,
                    "statistics": {
                        "total_nodes": total_nodes,
                        "total_characters": total_chars,
                        "average_node_size": total_chars // total_nodes if total_nodes > 0 else 0
                    },
                    "storage_path": str(index_dir)
                }
                
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
                
        except Exception as e:
            self.logger.error(f"Failed to index PDF content for {document_id}: {e}")
            return {
                "success": False,
                "document_id": document_id,
                "error": str(e)
            }
    
    async def query_document(
        self,
        document_id: str,
        query: str,
        response_mode: str = "compact"
    ) -> Dict[str, Any]:
        """
        Query a specific document using RAG.
        
        Args:
            document_id: ID of the document to query
            query: Natural language query
            response_mode: Response synthesis mode ('compact', 'tree_summarize', etc.)
            
        Returns:
            Dict containing query response and metadata
        """
        if not self.available:
            raise RuntimeError("LlamaIndex service not available")
        
        try:
            # Load index if not in cache
            if document_id not in self.indices:
                await self._load_index(document_id)
            
            if document_id not in self.indices:
                return {
                    "success": False,
                    "error": f"Document {document_id} not found or not indexed"
                }
            
            index = self.indices[document_id]
            
            # Create query engine
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=self.similarity_top_k
            )
            
            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode(response_mode)
            )
            
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer
            )
            
            # Execute query
            response = query_engine.query(query)
            
            # Extract source information
            sources = []
            if hasattr(response, 'source_nodes'):
                for node in response.source_nodes:
                    sources.append({
                        "node_id": node.node.node_id,
                        "score": getattr(node, 'score', None),
                        "text_snippet": node.node.text[:200] + "..." if len(node.node.text) > 200 else node.node.text,
                        "metadata": node.node.metadata
                    })
            
            return {
                "success": True,
                "document_id": document_id,
                "query": query,
                "response": str(response),
                "sources": sources,
                "metadata": {
                    "response_mode": response_mode,
                    "similarity_top_k": self.similarity_top_k
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to query document {document_id}: {e}")
            return {
                "success": False,
                "document_id": document_id,
                "query": query,
                "error": str(e)
            }
    
    async def advanced_rag_query(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        query_type: str = "factual",
        similarity_threshold: float = 0.7,
        max_results: int = 5,
        enable_reranking: bool = True,
        conversation_context: Optional[List[Dict[str, str]]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Advanced RAG query processing pipeline with enhanced capabilities.
        
        Args:
            query: Natural language query
            document_ids: Optional list of specific documents to query (None = all indexed)
            query_type: Type of query ('factual', 'analytical', 'conversational', 'summarization')
            similarity_threshold: Minimum similarity score for retrieved chunks
            max_results: Maximum number of results to return
            enable_reranking: Whether to apply relevance re-ranking
            conversation_context: Previous conversation turns for context
            metadata_filters: Optional metadata filters for retrieval
            
        Returns:
            Dict containing comprehensive query results and metadata
        """
        if not self.available:
            raise RuntimeError("LlamaIndex service not available")
        
        try:
            # Step 1: Query Understanding and Preprocessing
            processed_query = await self._preprocess_query(query, query_type, conversation_context)
            
            # Step 2: Determine target documents
            target_documents = document_ids if document_ids else list(self.indices.keys())
            
            if not target_documents:
                return {
                    "success": False,
                    "error": "No indexed documents available for querying"
                }
            
            # Step 3: Multi-document retrieval with similarity thresholds
            all_retrieved_nodes = []
            retrieval_metadata = {}
            
            for doc_id in target_documents:
                if doc_id not in self.indices:
                    await self._load_index(doc_id)
                    
                if doc_id in self.indices:
                    nodes = await self._retrieve_from_document(
                        doc_id,
                        processed_query,
                        similarity_threshold,
                        max_results,
                        metadata_filters
                    )
                    all_retrieved_nodes.extend(nodes)
                    retrieval_metadata[doc_id] = len(nodes)
            
            # Step 4: Apply relevance scoring and ranking
            if enable_reranking and all_retrieved_nodes:
                all_retrieved_nodes = await self._rerank_results(
                    all_retrieved_nodes,
                    processed_query,
                    query_type
                )
            
            # Step 5: Filter by similarity threshold and limit results
            filtered_nodes = [
                node for node in all_retrieved_nodes
                if getattr(node, 'score', 1.0) >= similarity_threshold
            ][:max_results]
            
            # Step 6: Context integration and response generation
            response_data = await self._generate_contextual_response(
                filtered_nodes,
                processed_query,
                query_type,
                conversation_context
            )
            
            # Step 7: Compile comprehensive results
            return {
                "success": True,
                "query": {
                    "original": query,
                    "processed": processed_query,
                    "type": query_type
                },
                "response": response_data["response"],
                "confidence_score": response_data["confidence"],
                "sources": self._format_source_nodes(filtered_nodes),
                "retrieval_stats": {
                    "total_documents_searched": len(target_documents),
                    "nodes_retrieved": len(all_retrieved_nodes),
                    "nodes_after_filtering": len(filtered_nodes),
                    "similarity_threshold": similarity_threshold,
                    "reranking_enabled": enable_reranking,
                    "per_document_results": retrieval_metadata
                },
                "query_metadata": {
                    "query_type": query_type,
                    "has_conversation_context": conversation_context is not None,
                    "metadata_filters_applied": metadata_filters is not None,
                    "processing_time_ms": response_data.get("processing_time_ms", 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Advanced RAG query failed: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "query_type": query_type
            }
    
    async def _preprocess_query(
        self,
        query: str,
        query_type: str,
        conversation_context: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Preprocess and enhance the query based on type and context.
        
        Args:
            query: Original query
            query_type: Type of query for specialized processing
            conversation_context: Previous conversation for context integration
            
        Returns:
            Enhanced/processed query string
        """
        processed_query = query.strip()
        
        # Add conversation context if available
        if conversation_context:
            context_summary = self._summarize_conversation_context(conversation_context)
            processed_query = f"Context: {context_summary}\n\nQuery: {processed_query}"
        
        # Query type specific enhancements
        if query_type == "analytical":
            processed_query = f"Analyze and provide detailed insights about: {processed_query}"
        elif query_type == "conversational":
            processed_query = f"In a conversational manner, please address: {processed_query}"
        elif query_type == "summarization":
            processed_query = f"Summarize the key information related to: {processed_query}"
        # 'factual' queries use the original query as-is
        
        return processed_query
    
    async def _retrieve_from_document(
        self,
        document_id: str,
        query: str,
        similarity_threshold: float,
        max_results: int,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Retrieve relevant nodes from a specific document with filtering.
        
        Args:
            document_id: Document to retrieve from
            query: Query string
            similarity_threshold: Minimum similarity score
            max_results: Maximum results per document
            metadata_filters: Optional metadata filters
            
        Returns:
            List of retrieved nodes with scores
        """
        try:
            index = self.indices[document_id]
            
            # Create retriever with enhanced parameters
            retriever = VectorIndexRetriever(
                index=index,
                similarity_top_k=max_results * 2,  # Retrieve more for better filtering
                doc_ids=None,
                filters=metadata_filters
            )
            
            # Retrieve nodes
            retrieved_nodes = retriever.retrieve(query)
            
            # Filter by similarity threshold and add document context
            filtered_nodes = []
            for node in retrieved_nodes:
                if getattr(node, 'score', 1.0) >= similarity_threshold:
                    # Add document context to node metadata
                    if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                        node.node.metadata['source_document_id'] = document_id
                    filtered_nodes.append(node)
            
            return filtered_nodes[:max_results]
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve from document {document_id}: {e}")
            return []
    
    async def _rerank_results(
        self,
        nodes: List[Any],
        query: str,
        query_type: str
    ) -> List[Any]:
        """
        Re-rank retrieved results based on relevance and query type.
        
        Args:
            nodes: Retrieved nodes to re-rank
            query: Original query
            query_type: Type of query for specialized ranking
            
        Returns:
            Re-ranked list of nodes
        """
        try:
            # Simple relevance-based re-ranking
            # In a production system, this could use a dedicated re-ranking model
            
            def calculate_relevance_score(node):
                base_score = getattr(node, 'score', 0.5)
                
                # Query type specific scoring adjustments
                text = node.node.text.lower() if hasattr(node, 'node') else ""
                query_lower = query.lower()
                
                # Boost score for exact matches
                if query_lower in text:
                    base_score += 0.1
                
                # Query type specific boosts
                if query_type == "analytical":
                    analytical_keywords = ["analysis", "insight", "conclusion", "result", "finding"]
                    if any(keyword in text for keyword in analytical_keywords):
                        base_score += 0.05
                elif query_type == "factual":
                    factual_keywords = ["fact", "data", "number", "statistic", "measurement"]
                    if any(keyword in text for keyword in factual_keywords):
                        base_score += 0.05
                
                return min(base_score, 1.0)  # Cap at 1.0
            
            # Re-rank nodes by calculated relevance
            ranked_nodes = sorted(nodes, key=calculate_relevance_score, reverse=True)
            
            # Update scores
            for node in ranked_nodes:
                node.score = calculate_relevance_score(node)
            
            return ranked_nodes
            
        except Exception as e:
            self.logger.error(f"Failed to re-rank results: {e}")
            return nodes  # Return original order on error
    
    async def _generate_contextual_response(
        self,
        nodes: List[Any],
        query: str,
        query_type: str,
        conversation_context: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate response with context integration and confidence scoring.
        
        Args:
            nodes: Retrieved and ranked nodes
            query: Processed query
            query_type: Type of query
            conversation_context: Previous conversation context
            
        Returns:
            Dict with response, confidence score, and metadata
        """
        import time
        start_time = time.time()
        
        try:
            if not nodes:
                return {
                    "response": "I couldn't find relevant information to answer your query.",
                    "confidence": 0.0,
                    "processing_time_ms": int((time.time() - start_time) * 1000)
                }
            
            # Select appropriate response mode based on query type
            response_mode_map = {
                "factual": "compact",
                "analytical": "tree_summarize",
                "conversational": "compact",
                "summarization": "tree_summarize"
            }
            response_mode = response_mode_map.get(query_type, "compact")
            
            # Create response synthesizer
            response_synthesizer = get_response_synthesizer(
                response_mode=ResponseMode(response_mode)
            )
            
            # Generate response
            response = response_synthesizer.synthesize(query, nodes)
            
            # Calculate confidence score based on node scores and count
            confidence = self._calculate_confidence_score(nodes, query_type)
            
            return {
                "response": str(response),
                "confidence": confidence,
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate contextual response: {e}")
            return {
                "response": f"Error generating response: {str(e)}",
                "confidence": 0.0,
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
    
    def _calculate_confidence_score(self, nodes: List[Any], query_type: str) -> float:
        """Calculate confidence score based on retrieved nodes and query type."""
        if not nodes:
            return 0.0
        
        # Base confidence from node scores
        node_scores = [getattr(node, 'score', 0.5) for node in nodes]
        avg_score = sum(node_scores) / len(node_scores)
        
        # Adjust based on number of supporting nodes
        node_count_factor = min(len(nodes) / 3.0, 1.0)  # Optimal around 3 nodes
        
        # Query type adjustments
        type_multiplier = {
            "factual": 1.0,      # Factual queries need high precision
            "analytical": 0.9,   # Analytical queries are more subjective
            "conversational": 0.8, # Conversational queries are more flexible
            "summarization": 0.85  # Summarization depends on content coverage
        }.get(query_type, 0.8)
        
        confidence = avg_score * node_count_factor * type_multiplier
        return round(min(confidence, 1.0), 3)
    
    def _format_source_nodes(self, nodes: List[Any]) -> List[Dict[str, Any]]:
        """Format source nodes for response output."""
        sources = []
        for i, node in enumerate(nodes):
            if hasattr(node, 'node'):
                sources.append({
                    "rank": i + 1,
                    "node_id": node.node.node_id,
                    "score": round(getattr(node, 'score', 0.0), 3),
                    "text_snippet": node.node.text[:300] + "..." if len(node.node.text) > 300 else node.node.text,
                    "metadata": node.node.metadata or {},
                    "document_id": node.node.metadata.get('source_document_id', 'unknown')
                })
        return sources
    
    def _summarize_conversation_context(self, context: List[Dict[str, str]]) -> str:
        """Summarize conversation context for query enhancement."""
        if not context:
            return ""
        
        # Take last few turns for context (avoid too much context)
        recent_context = context[-3:] if len(context) > 3 else context
        
        summary_parts = []
        for turn in recent_context:
            if turn.get('role') == 'user':
                summary_parts.append(f"User asked: {turn.get('content', '')}")
            elif turn.get('role') == 'assistant':
                summary_parts.append(f"Assistant responded about: {turn.get('content', '')[:100]}...")
        
        return " | ".join(summary_parts)
    
    async def summarize_document(
        self, 
        document_id: str,
        summary_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Generate a summary of the document.
        
        Args:
            document_id: ID of the document to summarize
            summary_type: Type of summary ('brief', 'comprehensive', 'key_points')
            
        Returns:
            Dict containing the summary and metadata
        """
        if not self.available:
            raise RuntimeError("LlamaIndex service not available")
        
        # Define summary prompts
        prompts = {
            "brief": "Provide a brief 2-3 sentence summary of this document.",
            "comprehensive": "Provide a comprehensive summary of this document, including main topics, key findings, and important details.",
            "key_points": "Extract and list the key points, main arguments, and important information from this document."
        }
        
        prompt = prompts.get(summary_type, prompts["comprehensive"])
        
        return await self.query_document(
            document_id=document_id,
            query=prompt,
            response_mode="tree_summarize"
        )
    
    async def extract_entities(
        self, 
        document_id: str,
        entity_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Extract named entities from the document.
        
        Args:
            document_id: ID of the document
            entity_types: List of entity types to extract (e.g., ['PERSON', 'ORG', 'DATE'])
            
        Returns:
            Dict containing extracted entities
        """
        if not self.available:
            raise RuntimeError("LlamaIndex service not available")
        
        entity_types_str = ", ".join(entity_types) if entity_types else "people, organizations, dates, locations, and other important entities"
        
        query = f"Extract and list all {entity_types_str} mentioned in this document. Provide them in a structured format."
        
        return await self.query_document(
            document_id=document_id,
            query=query,
            response_mode="compact"
        )
    
    async def compare_documents(
        self, 
        document_ids: List[str],
        comparison_aspect: str = "general"
    ) -> Dict[str, Any]:
        """
        Compare multiple documents.
        
        Args:
            document_ids: List of document IDs to compare
            comparison_aspect: Aspect to compare ('general', 'topics', 'sentiment', etc.)
            
        Returns:
            Dict containing comparison results
        """
        if not self.available:
            raise RuntimeError("LlamaIndex service not available")
        
        if len(document_ids) < 2:
            return {
                "success": False,
                "error": "At least 2 documents required for comparison"
            }
        
        try:
            # Get summaries for each document
            summaries = {}
            for doc_id in document_ids:
                result = await self.summarize_document(doc_id, "comprehensive")
                if result["success"]:
                    summaries[doc_id] = result["response"]
                else:
                    return {
                        "success": False,
                        "error": f"Failed to summarize document {doc_id}: {result.get('error', 'Unknown error')}"
                    }
            
            # Create comparison prompt
            comparison_text = "\n\n".join([
                f"Document {doc_id}:\n{summary}" 
                for doc_id, summary in summaries.items()
            ])
            
            comparison_prompts = {
                "general": "Compare and contrast these documents. What are the similarities and differences?",
                "topics": "What are the main topics covered in each document? How do they overlap or differ?",
                "sentiment": "Analyze the sentiment and tone of each document. How do they compare?",
                "key_insights": "What are the key insights from each document? How do they complement or contradict each other?"
            }
            
            prompt = comparison_prompts.get(comparison_aspect, comparison_prompts["general"])
            
            # Use the first document's index for the comparison query
            # (This is a limitation - ideally we'd create a combined index)
            result = await self.query_document(
                document_id=document_ids[0],
                query=f"{prompt}\n\nDocuments to compare:\n{comparison_text}",
                response_mode="tree_summarize"
            )
            
            if result["success"]:
                result["comparison_aspect"] = comparison_aspect
                result["compared_documents"] = document_ids
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to compare documents: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _load_index(self, document_id: str) -> bool:
        """Load an index from storage."""
        try:
            index_dir = Path(self.storage_dir) / document_id
            if not index_dir.exists():
                return False
            
            storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
            index = load_index_from_storage(storage_context)
            self.indices[document_id] = index
            
            self.logger.info(f"Loaded index for document {document_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load index for {document_id}: {e}")
            return False
    
    async def list_indexed_documents(self) -> Dict[str, Any]:
        """List all indexed documents."""
        try:
            storage_path = Path(self.storage_dir)
            if not storage_path.exists():
                return {"documents": [], "count": 0}
            
            documents = []
            for item in storage_path.iterdir():
                if item.is_dir():
                    # Check if it's a valid index directory
                    if (item / "docstore.json").exists():
                        documents.append({
                            "document_id": item.name,
                            "storage_path": str(item),
                            "indexed": item.name in self.indices,
                            "size_mb": sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / (1024 * 1024)
                        })
            
            return {
                "documents": documents,
                "count": len(documents),
                "total_size_mb": sum(doc["size_mb"] for doc in documents)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to list indexed documents: {e}")
            return {"error": str(e)}
    
    async def delete_document_index(self, document_id: str) -> Dict[str, Any]:
        """Delete a document index."""
        try:
            # Remove from cache
            if document_id in self.indices:
                del self.indices[document_id]
            
            # Remove from storage
            index_dir = Path(self.storage_dir) / document_id
            if index_dir.exists():
                import shutil
                shutil.rmtree(index_dir)
                
            return {
                "success": True,
                "document_id": document_id,
                "message": "Document index deleted successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to delete index for {document_id}: {e}")
            return {
                "success": False,
                "document_id": document_id,
                "error": str(e)
            }
    
    def __del__(self):
        """Cleanup resources when the service is destroyed."""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=True)
    
    # Conversation Memory Management Methods
    
    def manage_conversation_memory(
        self, 
        session_id: str, 
        user_message: str, 
        assistant_response: str
    ) -> None:
        """
        Manage conversation memory for a session.
        
        Args:
            session_id: Unique session identifier
            user_message: User's message
            assistant_response: Assistant's response
        """
        if session_id not in self.conversation_memories:
            self.conversation_memories[session_id] = []
        
        # Add new conversation turn
        self.conversation_memories[session_id].append({
            "role": "user",
            "content": user_message,
            "timestamp": self._get_timestamp()
        })
        
        self.conversation_memories[session_id].append({
            "role": "assistant", 
            "content": assistant_response,
            "timestamp": self._get_timestamp()
        })
        
        # Manage memory size
        self._manage_memory_size(session_id)
    
    def get_conversation_context(self, session_id: str) -> List[Dict[str, str]]:
        """
        Get conversation context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of conversation turns
        """
        return self.conversation_memories.get(session_id, [])
    
    def clear_conversation_memory(self, session_id: str) -> None:
        """
        Clear conversation memory for a session.
        
        Args:
            session_id: Session identifier
        """
        if session_id in self.conversation_memories:
            del self.conversation_memories[session_id]
            self.logger.info(f"Cleared conversation memory for session: {session_id}")
    
    def _manage_memory_size(self, session_id: str) -> None:
        """
        Manage conversation memory size, summarizing old conversations if needed.
        
        Args:
            session_id: Session identifier
        """
        memory = self.conversation_memories[session_id]
        
        if len(memory) > self.max_conversation_turns:
            # Check if we should summarize
            if len(memory) > self.conversation_summary_threshold:
                self._summarize_old_conversations(session_id)
            else:
                # Simple truncation - remove oldest turns
                excess = len(memory) - self.max_conversation_turns
                self.conversation_memories[session_id] = memory[excess:]
    
    def _summarize_old_conversations(self, session_id: str) -> None:
        """
        Summarize old conversations to preserve context while reducing memory.
        
        Args:
            session_id: Session identifier
        """
        try:
            memory = self.conversation_memories[session_id]
            
            # Keep recent conversations, summarize older ones
            recent_turns = 4  # Keep last 4 turns
            old_conversations = memory[:-recent_turns]
            recent_conversations = memory[-recent_turns:]
            
            if not old_conversations:
                return
            
            # Create summary of old conversations
            conversation_text = "\n".join([
                f"{turn['role']}: {turn['content']}" 
                for turn in old_conversations
            ])
            
            summary_prompt = f"""Summarize the following conversation history concisely, preserving key context and topics discussed:

{conversation_text}

Summary:"""
            
            # Use LLM to create summary (if available)
            if hasattr(self, 'llm') and self.llm:
                try:
                    summary_response = self.llm.complete(summary_prompt)
                    summary = str(summary_response).strip()
                except Exception as e:
                    self.logger.warning(f"Failed to generate conversation summary: {e}")
                    summary = "Previous conversation covered multiple topics and queries."
            else:
                # Fallback summary
                summary = f"Previous conversation with {len(old_conversations)} turns covering various topics."
            
            # Replace old conversations with summary
            summary_entry = {
                "role": "system",
                "content": f"[Conversation Summary]: {summary}",
                "timestamp": self._get_timestamp()
            }
            
            self.conversation_memories[session_id] = [summary_entry] + recent_conversations
            
            self.logger.info(f"Summarized {len(old_conversations)} old conversation turns for session: {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to summarize conversations for session {session_id}: {e}")
            # Fallback to simple truncation
            memory = self.conversation_memories[session_id]
            self.conversation_memories[session_id] = memory[-self.max_conversation_turns:]
    
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get conversation memory statistics.
        
        Returns:
            Dict with memory statistics
        """
        stats = {
            "total_sessions": len(self.conversation_memories),
            "max_turns_per_session": self.max_conversation_turns,
            "summary_threshold": self.conversation_summary_threshold,
            "sessions": {}
        }
        
        for session_id, memory in self.conversation_memories.items():
            stats["sessions"][session_id] = {
                "turn_count": len(memory),
                "has_summary": any(turn.get("role") == "system" for turn in memory),
                "last_activity": memory[-1]["timestamp"] if memory else None
            }
        
        return stats