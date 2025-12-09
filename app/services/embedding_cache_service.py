"""
Embedding Cache Service for RAG Optimization

This service implements Phase 1 RAG optimization: Embedding Caching & Precomputation.
Provides in-memory and Redis-based caching for embeddings to reduce API costs and improve performance.
"""

import logging
import hashlib
import pickle
from typing import Optional, Dict, Any
import numpy as np
from datetime import datetime, timedelta

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available - embedding cache will use memory only")


class EmbeddingCacheService:
    """
    Embedding cache service with in-memory and Redis persistence.
    
    Features:
    - In-memory LRU cache for fast access
    - Redis persistence for shared cache across instances
    - Configurable TTL and max size
    - Automatic cache eviction
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl: int = 86400,  # 24 hours default
        max_size: int = 10000,
        enabled: bool = True
    ):
        """
        Initialize embedding cache service.
        
        Args:
            redis_url: Redis connection URL (optional)
            ttl: Time-to-live for cached embeddings in seconds
            max_size: Maximum number of embeddings to cache in memory
            enabled: Whether caching is enabled
        """
        self.logger = logging.getLogger(__name__)
        self.enabled = enabled
        self.ttl = ttl
        self.max_size = max_size
        
        # In-memory cache
        self.cache: Dict[str, tuple] = {}  # {cache_key: (embedding, timestamp)}
        self.access_count: Dict[str, int] = {}  # For LRU eviction
        
        # Redis cache
        self.redis_client = None
        if REDIS_AVAILABLE and redis_url and enabled:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=False)
                self.redis_client.ping()
                self.logger.info("Redis cache initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Redis cache: {e}")
                self.redis_client = None
    
    def _generate_cache_key(self, text: str, model: str) -> str:
        """Generate cache key from text and model."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() - timestamp > timedelta(seconds=self.ttl)
    
    def _evict_lru(self):
        """Evict least recently used item if cache is full."""
        if len(self.cache) >= self.max_size:
            # Find least recently used key
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]
            self.logger.debug(f"Evicted LRU cache entry: {lru_key[:16]}...")
    
    async def get(self, text: str, model: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache.
        
        Args:
            text: Text to get embedding for
            model: Model name used for embedding
            
        Returns:
            Cached embedding or None if not found
        """
        if not self.enabled:
            return None
        
        try:
            cache_key = self._generate_cache_key(text, model)
            
            # Check memory cache first
            if cache_key in self.cache:
                embedding, timestamp = self.cache[cache_key]
                
                # Check if expired
                if self._is_expired(timestamp):
                    del self.cache[cache_key]
                    del self.access_count[cache_key]
                else:
                    # Update access count
                    self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
                    self.logger.debug(f"Cache hit (memory): {cache_key[:16]}...")
                    return embedding
            
            # Check Redis cache
            if self.redis_client:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    embedding = pickle.loads(cached_data)
                    
                    # Store in memory cache for faster access
                    self._evict_lru()
                    self.cache[cache_key] = (embedding, datetime.now())
                    self.access_count[cache_key] = 1
                    
                    self.logger.debug(f"Cache hit (Redis): {cache_key[:16]}...")
                    return embedding
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get from cache: {e}")
            return None
    
    async def set(self, text: str, model: str, embedding: np.ndarray):
        """
        Store embedding in cache.
        
        Args:
            text: Text that was embedded
            model: Model name used for embedding
            embedding: Embedding vector to cache
        """
        if not self.enabled:
            return
        
        try:
            cache_key = self._generate_cache_key(text, model)
            
            # Store in memory cache
            self._evict_lru()
            self.cache[cache_key] = (embedding, datetime.now())
            self.access_count[cache_key] = 1
            
            # Store in Redis cache
            if self.redis_client:
                serialized = pickle.dumps(embedding)
                self.redis_client.setex(cache_key, self.ttl, serialized)
            
            self.logger.debug(f"Cached embedding: {cache_key[:16]}...")
            
        except Exception as e:
            self.logger.error(f"Failed to set cache: {e}")

