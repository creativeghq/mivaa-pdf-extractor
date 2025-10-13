"""
Embedding Service for LlamaIndex RAG Implementation

This service provides centralized embedding management for the PDF2Markdown microservice,
including embedding generation, caching, batch processing, and model management.
Designed to work seamlessly with the existing LlamaIndex service and SupabaseVectorStore.
"""

import logging
import os
import asyncio
import hashlib
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
import numpy as np

try:
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core.embeddings import BaseEmbedding
    import tiktoken
    EMBEDDING_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Embedding dependencies not available: {e}")
    EMBEDDING_DEPENDENCIES_AVAILABLE = False


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models and processing."""
    model_name: str = "text-embedding-ada-002"  # PLATFORM STANDARD
    model_type: str = "openai"  # "openai" or "huggingface"
    dimension: int = 1536
    batch_size: int = 100
    max_tokens: int = 8191  # OpenAI text-embedding-ada-002 limit
    cache_enabled: bool = True
    cache_ttl_hours: int = 24
    rate_limit_rpm: int = 3000  # Requests per minute
    rate_limit_tpm: int = 1000000  # Tokens per minute


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    text: str
    embedding: List[float]
    model_name: str
    dimension: int
    token_count: int
    processing_time: float
    cached: bool = False
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class BatchEmbeddingResult:
    """Result of batch embedding generation."""
    results: List[EmbeddingResult]
    total_texts: int
    successful_embeddings: int
    failed_embeddings: int
    total_tokens: int
    total_processing_time: float
    cache_hits: int
    cache_misses: int


class EmbeddingCache:
    """Simple in-memory cache for embeddings with TTL support."""
    
    def __init__(self, ttl_hours: int = 24):
        self.cache: Dict[str, Tuple[List[float], datetime]] = {}
        self.ttl = timedelta(hours=ttl_hours)
        self.logger = logging.getLogger(__name__)
    
    def _generate_key(self, text: str, model_name: str) -> str:
        """Generate cache key from text and model name."""
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get embedding from cache if available and not expired."""
        key = self._generate_key(text, model_name)
        
        if key in self.cache:
            embedding, created_at = self.cache[key]
            if datetime.utcnow() - created_at < self.ttl:
                return embedding
            else:
                # Remove expired entry
                del self.cache[key]
        
        return None
    
    def set(self, text: str, model_name: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        key = self._generate_key(text, model_name)
        self.cache[key] = (embedding, datetime.utcnow())
    
    def clear_expired(self) -> int:
        """Remove expired entries from cache."""
        current_time = datetime.utcnow()
        expired_keys = [
            key for key, (_, created_at) in self.cache.items()
            if current_time - created_at >= self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            self.logger.info(f"Cleared {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_entries": len(self.cache),
            "memory_usage_mb": sum(
                len(json.dumps(embedding).encode()) 
                for embedding, _ in self.cache.values()
            ) / (1024 * 1024)
        }


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, requests_per_minute: int = 3000, tokens_per_minute: int = 1000000):
        self.rpm_limit = requests_per_minute
        self.tpm_limit = tokens_per_minute
        self.request_times: List[datetime] = []
        self.token_usage: List[Tuple[datetime, int]] = []
        self.logger = logging.getLogger(__name__)
    
    async def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        """Wait if rate limits would be exceeded."""
        current_time = datetime.utcnow()
        one_minute_ago = current_time - timedelta(minutes=1)
        
        # Clean old entries
        self.request_times = [t for t in self.request_times if t > one_minute_ago]
        self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > one_minute_ago]
        
        # Check request rate limit
        if len(self.request_times) >= self.rpm_limit:
            wait_time = 60 - (current_time - self.request_times[0]).total_seconds()
            if wait_time > 0:
                self.logger.info(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Check token rate limit
        current_tokens = sum(tokens for _, tokens in self.token_usage)
        if current_tokens + estimated_tokens > self.tpm_limit:
            wait_time = 60 - (current_time - self.token_usage[0][0]).total_seconds()
            if wait_time > 0:
                self.logger.info(f"Token rate limit reached, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.request_times.append(current_time)
        if estimated_tokens > 0:
            self.token_usage.append((current_time, estimated_tokens))


class EmbeddingService:
    """
    Centralized embedding service for LlamaIndex RAG implementation.
    
    Features:
    - Multiple embedding model support (OpenAI, HuggingFace)
    - Intelligent caching with TTL
    - Batch processing for efficiency
    - Rate limiting for API compliance
    - Token counting and optimization
    - Performance monitoring and metrics
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """Initialize the embedding service with configuration."""
        self.config = config or EmbeddingConfig()
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Service availability
        self.available = EMBEDDING_DEPENDENCIES_AVAILABLE
        if not self.available:
            self.logger.warning("Embedding service unavailable - dependencies not installed")
            return
        
        # Initialize components
        self.embedding_model: Optional[BaseEmbedding] = None
        self.tokenizer = None
        self.cache = EmbeddingCache(ttl_hours=self.config.cache_ttl // 3600) if self.config.enable_cache else None
        self.rate_limiter = RateLimiter(
            requests_per_minute=self.config.rate_limit_rpm,
            tokens_per_minute=self.config.rate_limit_tpm
        )
        
        # Performance metrics
        self.metrics = {
            "total_embeddings_generated": 0,
            "total_tokens_processed": 0,
            "total_processing_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "batch_operations": 0,
            "rate_limit_waits": 0
        }
        
        # Initialize embedding model
        self._initialize_embedding_model()
        
        self.logger.info(f"Embedding service initialized with model: {self.config.model_name}")
    
    def _initialize_embedding_model(self) -> None:
        """Initialize the embedding model based on configuration."""
        if not self.available:
            return
        
        try:
            if self.config.model_type.lower() == "openai":
                self.embedding_model = OpenAIEmbedding(
                    model=self.config.model_name,
                    api_key=os.getenv('OPENAI_API_KEY'),
                    dimensions=self.config.dimension if self.config.model_name.startswith('text-embedding-3') else None
                )
                
                # Initialize tokenizer for token counting
                try:
                    self.tokenizer = tiktoken.encoding_for_model(self.config.model_name)
                except KeyError:
                    # Fallback for custom models
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                    
            elif self.config.model_type.lower() == "huggingface":
                self.embedding_model = HuggingFaceEmbedding(
                    model_name=self.config.model_name,
                    max_length=self.config.max_tokens
                )
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            self.logger.info(f"Initialized {self.config.model_type} embedding model: {self.config.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            self.available = False
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the appropriate tokenizer."""
        if not self.tokenizer:
            # Rough estimation: ~4 characters per token
            return len(text) // 4
        
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            self.logger.warning(f"Token counting failed: {e}")
            return len(text) // 4
    
    def truncate_text(self, text: str, max_tokens: Optional[int] = None) -> str:
        """Truncate text to fit within token limits."""
        max_tokens = max_tokens or self.config.max_tokens
        
        if not self.tokenizer:
            # Rough truncation based on character count
            max_chars = max_tokens * 4
            return text[:max_chars] if len(text) > max_chars else text
        
        try:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
            
            # Truncate and decode
            truncated_tokens = tokens[:max_tokens]
            return self.tokenizer.decode(truncated_tokens)
            
        except Exception as e:
            self.logger.warning(f"Text truncation failed: {e}")
            # Fallback to character-based truncation
            max_chars = max_tokens * 4
            return text[:max_chars] if len(text) > max_chars else text
    
    async def generate_embedding(
        self, 
        text: str, 
        use_cache: bool = True,
        truncate_if_needed: bool = True
    ) -> EmbeddingResult:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            use_cache: Whether to use caching
            truncate_if_needed: Whether to truncate text if it exceeds token limits
            
        Returns:
            EmbeddingResult with embedding and metadata
        """
        if not self.available:
            raise RuntimeError("Embedding service not available")
        
        start_time = datetime.utcnow()
        
        # Truncate text if needed
        if truncate_if_needed:
            text = self.truncate_text(text)
        
        # Count tokens
        token_count = self.count_tokens(text)
        
        # Check cache first
        cached_embedding = None
        if use_cache and self.cache:
            cached_embedding = self.cache.get(text, self.config.model_name)
            if cached_embedding:
                self.metrics["cache_hits"] += 1
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                return EmbeddingResult(
                    text=text,
                    embedding=cached_embedding,
                    model_name=self.config.model_name,
                    dimension=len(cached_embedding),
                    token_count=token_count,
                    processing_time=processing_time,
                    cached=True,
                    created_at=start_time
                )
        
        # Rate limiting
        await self.rate_limiter.wait_if_needed(token_count)
        
        # Generate embedding
        try:
            embedding = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.embedding_model.get_text_embedding(text)
            )
            
            # Cache the result
            if use_cache and self.cache:
                self.cache.set(text, self.config.model_name, embedding)
                self.metrics["cache_misses"] += 1
            
            # Update metrics
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.metrics["total_embeddings_generated"] += 1
            self.metrics["total_tokens_processed"] += token_count
            self.metrics["total_processing_time"] += processing_time
            
            return EmbeddingResult(
                text=text,
                embedding=embedding,
                model_name=self.config.model_name,
                dimension=len(embedding),
                token_count=token_count,
                processing_time=processing_time,
                cached=False,
                created_at=start_time
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def generate_embeddings_batch(
        self, 
        texts: List[str],
        use_cache: bool = True,
        truncate_if_needed: bool = True,
        batch_size: Optional[int] = None
    ) -> BatchEmbeddingResult:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use caching
            truncate_if_needed: Whether to truncate texts if they exceed token limits
            batch_size: Override default batch size
            
        Returns:
            BatchEmbeddingResult with all embeddings and statistics
        """
        if not self.available:
            raise RuntimeError("Embedding service not available")
        
        start_time = datetime.utcnow()
        batch_size = batch_size or self.config.batch_size
        
        results: List[EmbeddingResult] = []
        successful_embeddings = 0
        failed_embeddings = 0
        total_tokens = 0
        cache_hits = 0
        cache_misses = 0
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [
                self.generate_embedding(
                    text=text,
                    use_cache=use_cache,
                    truncate_if_needed=truncate_if_needed
                )
                for text in batch_texts
            ]
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, EmbeddingResult):
                        results.append(result)
                        successful_embeddings += 1
                        total_tokens += result.token_count
                        if result.cached:
                            cache_hits += 1
                        else:
                            cache_misses += 1
                    else:
                        failed_embeddings += 1
                        self.logger.error(f"Batch embedding failed: {result}")
                        
            except Exception as e:
                self.logger.error(f"Batch processing failed: {e}")
                failed_embeddings += len(batch_texts)
        
        # Update metrics
        total_processing_time = (datetime.utcnow() - start_time).total_seconds()
        self.metrics["batch_operations"] += 1
        
        return BatchEmbeddingResult(
            results=results,
            total_texts=len(texts),
            successful_embeddings=successful_embeddings,
            failed_embeddings=failed_embeddings,
            total_tokens=total_tokens,
            total_processing_time=total_processing_time,
            cache_hits=cache_hits,
            cache_misses=cache_misses
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the embedding service."""
        if not self.available:
            return {
                "status": "unavailable",
                "message": "Embedding service dependencies not installed",
                "model": None,
                "cache": None
            }
        
        try:
            # Test embedding generation with a simple text
            test_result = await self.generate_embedding("test", use_cache=False)
            
            cache_stats = self.cache.get_stats() if self.cache else None
            
            return {
                "status": "healthy",
                "message": "Embedding service operational",
                "model": {
                    "name": self.config.model_name,
                    "type": self.config.model_type,
                    "dimension": self.config.dimension,
                    "max_tokens": self.config.max_tokens
                },
                "cache": {
                    "enabled": self.config.enable_cache,
                    "stats": cache_stats
                },
                "metrics": self.metrics,
                "test_embedding": {
                    "dimension": test_result.dimension,
                    "processing_time": test_result.processing_time,
                    "token_count": test_result.token_count
                }
            }
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "message": f"Health check failed: {str(e)}",
                "model": {
                    "name": self.config.model_name,
                    "type": self.config.model_type
                },
                "cache": {"enabled": self.config.enable_cache}
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service performance metrics."""
        cache_stats = self.cache.get_stats() if self.cache else {}
        
        return {
            "embeddings": self.metrics,
            "cache": cache_stats,
            "config": asdict(self.config)
        }
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear the embedding cache."""
        if not self.cache:
            return {"message": "Cache not enabled"}
        
        entries_before = len(self.cache.cache)
        self.cache.cache.clear()
        
        return {
            "message": f"Cache cleared, removed {entries_before} entries",
            "entries_removed": entries_before
        }
    
    def cleanup_cache(self) -> Dict[str, Any]:
        """Clean up expired cache entries."""
        if not self.cache:
            return {"message": "Cache not enabled"}
        
        expired_count = self.cache.clear_expired()
        
        return {
            "message": f"Cache cleanup completed",
            "expired_entries_removed": expired_count,
            "remaining_entries": len(self.cache.cache)
        }
    
    def __del__(self):
        """Cleanup resources when the service is destroyed."""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=True)