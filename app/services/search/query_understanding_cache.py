"""
Query Understanding Cache

Caches the LLM-based query parser output (Qwen / GPT-5-mini) so that repeat
queries skip the ~2-5 second LLM call entirely.

The parse output is deterministic for a given query (system prompt is fixed,
temperature 0.1) so caching is safe and the only invalidation needed is when
the system prompt itself changes.

Storage: Postgres table `query_understanding_cache`, keyed on sha256 of the
normalised query text.
"""

import hashlib
import logging
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class QueryUnderstandingCache:
    """Cache for LLM query parser output."""

    def __init__(self):
        self.supabase = get_supabase_client()
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _hash_query(query: str) -> str:
        """Normalise + hash the query for cache key."""
        normalised = query.strip().lower()
        return hashlib.sha256(normalised.encode("utf-8")).hexdigest()

    async def lookup(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Look up a cached parse for this query.

        Returns the cached row dict if found, None otherwise.
        Increments hit_count and last_hit_at on a hit (fire-and-forget).
        """
        try:
            query_hash = self._hash_query(query)
            result = (
                self.supabase.client.table("query_understanding_cache")
                .select("*")
                .eq("query_hash", query_hash)
                .limit(1)
                .execute()
            )

            if result.data and len(result.data) > 0:
                row = result.data[0]
                # Bump hit counter (fire-and-forget — don't block the search)
                try:
                    self.supabase.client.table("query_understanding_cache").update(
                        {
                            "hit_count": (row.get("hit_count") or 0) + 1,
                            "last_hit_at": datetime.utcnow().isoformat(),
                        }
                    ).eq("query_hash", query_hash).execute()
                except Exception as bump_err:
                    self.logger.debug(f"Cache hit-counter bump failed: {bump_err}")

                self.logger.info(f"💾 Query cache HIT: '{query[:50]}'")
                return row

            return None

        except Exception as e:
            # Cache miss should never break the search
            self.logger.warning(f"Query cache lookup failed: {e}")
            return None

    async def store(
        self,
        query: str,
        parsed_data: Dict[str, Any],
        visual_query: str,
        filters: Dict[str, Any],
        weight_profile: str,
        dynamic_weights: Dict[str, float],
        is_product_name: bool,
        model_used: str,
        parse_latency_ms: int,
    ) -> None:
        """Store a fresh parse in the cache. Idempotent on query_hash."""
        try:
            query_hash = self._hash_query(query)
            self.supabase.client.table("query_understanding_cache").upsert(
                {
                    "query_hash": query_hash,
                    "query_text": query,
                    "parsed_data": parsed_data,
                    "visual_query": visual_query,
                    "filters": filters,
                    "weight_profile": weight_profile,
                    "dynamic_weights": dynamic_weights,
                    "is_product_name": is_product_name,
                    "model_used": model_used,
                    "parse_latency_ms": parse_latency_ms,
                    "hit_count": 0,
                    "created_at": datetime.utcnow().isoformat(),
                },
                on_conflict="query_hash",
            ).execute()

            self.logger.debug(f"💾 Cached query parse: '{query[:50]}'")

        except Exception as e:
            # Storing should never break the search
            self.logger.warning(f"Query cache store failed: {e}")


# Singleton accessor
_cache_instance: Optional[QueryUnderstandingCache] = None


def get_query_understanding_cache() -> QueryUnderstandingCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = QueryUnderstandingCache()
    return _cache_instance
