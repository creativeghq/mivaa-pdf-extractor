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

try:
    from llama_index.core import (
        VectorStoreIndex, 
        Document, 
        Settings,
        StorageContext,
        load_index_from_storage
    )
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core.llms import LLM
    from llama_index.llms.openai import OpenAI
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.retrievers import VectorIndexRetriever
    from llama_index.core.response_synthesizers import ResponseMode
    from llama_index.core.response_synthesizers import get_response_synthesizer
    from llama_index.readers.file import PDFReader
    LLAMAINDEX_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LlamaIndex not available: {e}")
    LLAMAINDEX_AVAILABLE = False


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
        self.embedding_model = self.config.get('embedding_model', 'text-embedding-ada-002')
        self.llm_model = self.config.get('llm_model', 'gpt-3.5-turbo')
        self.chunk_size = self.config.get('chunk_size', 1024)
        self.chunk_overlap = self.config.get('chunk_overlap', 200)
        self.similarity_top_k = self.config.get('similarity_top_k', 5)
        
        # Storage
        self.storage_dir = self.config.get('storage_dir', tempfile.mkdtemp(prefix='llamaindex_'))
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        # Index cache
        self.indices: Dict[str, VectorStoreIndex] = {}
        
        self.logger.info(f"LlamaIndex service initialized with storage: {self.storage_dir}")
    
    def _initialize_components(self):
        """Initialize LlamaIndex components."""
        if not self.available:
            return
        
        try:
            # Initialize embeddings
            if self.embedding_model.startswith('text-embedding'):
                # OpenAI embeddings
                self.embeddings = OpenAIEmbedding(
                    model=self.embedding_model,
                    api_key=os.getenv('OPENAI_API_KEY')
                )
            else:
                # HuggingFace embeddings as fallback
                self.embeddings = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
            
            # Initialize LLM
            if self.llm_model.startswith('gpt'):
                self.llm = OpenAI(
                    model=self.llm_model,
                    api_key=os.getenv('OPENAI_API_KEY'),
                    temperature=0.1
                )
            else:
                # Fallback to default
                self.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
            
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
                
                # Create index
                index = VectorStoreIndex.from_documents(
                    documents,
                    node_parser=self.node_parser
                )
                
                # Store index
                index_dir = Path(self.storage_dir) / document_id
                index_dir.mkdir(exist_ok=True)
                index.storage_context.persist(persist_dir=str(index_dir))
                
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