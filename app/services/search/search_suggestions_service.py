"""
Search Suggestions Service

This service provides intelligent search suggestions, auto-complete, trending searches,
typo correction, and query expansion functionality.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import re
from collections import defaultdict

from ..core.supabase_client import SupabaseClient
from ..schemas.suggestions import (
    SearchSuggestion,
    TrendingSearch,
    QueryCorrection,
    ExpandedQuery,
    PopularSearch
)
from ..utils.text_similarity import calculate_string_similarity

logger = logging.getLogger(__name__)


class SearchSuggestionsService:
    """Service for managing search suggestions and auto-complete."""
    
    def __init__(self, supabase_client: SupabaseClient):
        self.supabase = supabase_client
        self.client = supabase_client.client
        
        # Common material-related synonyms
        self.synonyms_map = {
            "fire": ["flame", "heat", "thermal"],
            "water": ["moisture", "liquid", "aqua"],
            "resistant": ["proof", "repellent", "protective"],
            "tile": ["tiles", "tiling", "ceramic"],
            "wood": ["wooden", "timber", "lumber"],
            "stone": ["rock", "marble", "granite"],
            "floor": ["flooring", "ground", "surface"],
            "wall": ["walls", "partition", "surface"],
        }
        
        # Common abbreviations
        self.abbreviations = {
            "res": "resistant",
            "cert": "certificate",
            "spec": "specification",
            "dim": "dimension",
            "mat": "material",
        }
    
    async def get_autocomplete_suggestions(
        self,
        query: str,
        limit: int = 10,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        include_trending: bool = True,
        include_recent: bool = True,
        include_popular: bool = True,
        categories: Optional[List[str]] = None
    ) -> Tuple[List[SearchSuggestion], Dict[str, Any]]:
        """
        Get auto-complete suggestions for a partial query.
        
        Returns:
            Tuple of (suggestions list, metadata dict)
        """
        start_time = datetime.now()
        suggestions = []
        metadata = {
            "sources": [],
            "query_length": len(query),
            "filters_applied": []
        }
        
        try:
            # 1. Get matching suggestions from database
            db_suggestions = await self._get_database_suggestions(query, limit, categories)
            suggestions.extend(db_suggestions)
            if db_suggestions:
                metadata["sources"].append("database")
            
            # 2. Get trending searches if enabled
            if include_trending and len(suggestions) < limit:
                trending = await self._get_trending_matches(query, limit - len(suggestions))
                suggestions.extend(trending)
                if trending:
                    metadata["sources"].append("trending")
            
            # 3. Get recent searches for user if enabled
            if include_recent and user_id and len(suggestions) < limit:
                recent = await self._get_recent_user_searches(user_id, query, limit - len(suggestions))
                suggestions.extend(recent)
                if recent:
                    metadata["sources"].append("recent")
            
            # 4. Get popular searches if enabled
            if include_popular and len(suggestions) < limit:
                popular = await self._get_popular_matches(query, limit - len(suggestions))
                suggestions.extend(popular)
                if popular:
                    metadata["sources"].append("popular")
            
            # 5. Get product/material name matches
            if len(suggestions) < limit:
                product_matches = await self._get_product_matches(query, limit - len(suggestions))
                suggestions.extend(product_matches)
                if product_matches:
                    metadata["sources"].append("products")
            
            # Remove duplicates while preserving order
            seen = set()
            unique_suggestions = []
            for sug in suggestions:
                if sug.suggestion_text.lower() not in seen:
                    seen.add(sug.suggestion_text.lower())
                    unique_suggestions.append(sug)
            
            # Sort by relevance (popularity_score)
            unique_suggestions.sort(key=lambda x: x.popularity_score, reverse=True)
            
            # Limit to requested number
            final_suggestions = unique_suggestions[:limit]
            
            # Track impressions
            if session_id:
                await self._track_impressions(final_suggestions, session_id)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            metadata["processing_time_ms"] = int(processing_time)
            metadata["total_found"] = len(final_suggestions)
            
            return final_suggestions, metadata
            
        except Exception as e:
            logger.error(f"Error getting autocomplete suggestions: {e}")
            return [], {"error": str(e)}
    
    async def _get_database_suggestions(
        self,
        query: str,
        limit: int,
        categories: Optional[List[str]] = None
    ) -> List[SearchSuggestion]:
        """Get suggestions from search_suggestions table."""
        try:
            query_builder = self.client.table("search_suggestions") \
                .select("*") \
                .eq("is_active", True) \
                .ilike("suggestion_text", f"{query}%") \
                .order("popularity_score", desc=True) \
                .limit(limit)
            
            if categories:
                query_builder = query_builder.in_("category", categories)
            
            response = query_builder.execute()
            
            if response.data:
                return [
                    SearchSuggestion(
                        id=row["id"],
                        suggestion_text=row["suggestion_text"],
                        suggestion_type=row["suggestion_type"],
                        category=row.get("category"),
                        popularity_score=float(row.get("popularity_score", 0)),
                        click_count=row.get("click_count", 0),
                        impression_count=row.get("impression_count", 0),
                        ctr=float(row.get("ctr", 0)),
                        metadata=row.get("metadata", {})
                    )
                    for row in response.data
                ]
            return []
        except Exception as e:
            logger.error(f"Error getting database suggestions: {e}")
            return []
    
    async def _get_trending_matches(self, query: str, limit: int) -> List[SearchSuggestion]:
        """Get trending searches that match the query."""
        try:
            response = self.client.table("trending_searches") \
                .select("*") \
                .ilike("query_text", f"%{query}%") \
                .eq("time_window", "daily") \
                .order("trend_score", desc=True) \
                .limit(limit) \
                .execute()
            
            if response.data:
                return [
                    SearchSuggestion(
                        id=row["id"],
                        suggestion_text=row["query_text"],
                        suggestion_type="trending",
                        category=row.get("category"),
                        popularity_score=min(float(row.get("trend_score", 0)) / 100, 1.0),
                        click_count=row.get("search_count", 0),
                        impression_count=row.get("search_count", 0),
                        ctr=0.5,  # Estimate for trending
                        metadata={
                            "growth_rate": row.get("growth_rate", 0),
                            "unique_users": row.get("unique_users_count", 0)
                        }
                    )
                    for row in response.data
                ]
            return []
        except Exception as e:
            logger.error(f"Error getting trending matches: {e}")
            return []
    
    async def _get_recent_user_searches(
        self,
        user_id: str,
        query: str,
        limit: int
    ) -> List[SearchSuggestion]:
        """Get user's recent searches that match the query."""
        try:
            response = self.client.table("search_analytics") \
                .select("query_text, created_at") \
                .eq("user_id", user_id) \
                .ilike("query_text", f"%{query}%") \
                .order("created_at", desc=True) \
                .limit(limit) \
                .execute()
            
            if response.data:
                return [
                    SearchSuggestion(
                        id=f"recent_{i}",
                        suggestion_text=row["query_text"],
                        suggestion_type="recent",
                        popularity_score=0.7,  # Medium priority for recent
                        click_count=1,
                        impression_count=1,
                        ctr=1.0,
                        metadata={"last_searched": row["created_at"]}
                    )
                    for i, row in enumerate(response.data)
                ]
            return []
        except Exception as e:
            logger.error(f"Error getting recent user searches: {e}")
            return []
    
    async def _get_popular_matches(self, query: str, limit: int) -> List[SearchSuggestion]:
        """Get popular searches that match the query."""
        try:
            # Get popular searches from analytics
            response = self.client.rpc(
                "get_popular_searches",
                {
                    "p_query_filter": query,
                    "p_limit": limit,
                    "p_days": 30
                }
            ).execute()
            
            if response.data:
                return [
                    SearchSuggestion(
                        id=f"popular_{i}",
                        suggestion_text=row["query_text"],
                        suggestion_type="popular",
                        popularity_score=min(float(row.get("search_count", 0)) / 100, 1.0),
                        click_count=row.get("search_count", 0),
                        impression_count=row.get("search_count", 0),
                        ctr=0.6,
                        metadata={
                            "unique_users": row.get("unique_users", 0),
                            "avg_results": row.get("avg_results", 0)
                        }
                    )
                    for i, row in enumerate(response.data)
                ]
            return []
        except Exception as e:
            logger.warning(f"Popular searches function not available: {e}")
            return []
    
    async def _get_product_matches(self, query: str, limit: int) -> List[SearchSuggestion]:
        """Get product/material name matches."""
        try:
            response = self.client.table("products") \
                .select("id, name, metadata") \
                .ilike("name", f"%{query}%") \
                .limit(limit) \
                .execute()
            
            if response.data:
                return [
                    SearchSuggestion(
                        id=row["id"],
                        suggestion_text=row["name"],
                        suggestion_type="product",
                        category=row.get("metadata", {}).get("category"),
                        popularity_score=0.8,
                        click_count=0,
                        impression_count=0,
                        ctr=0.0,
                        metadata={"product_id": row["id"]}
                    )
                    for row in response.data
                ]
            return []
        except Exception as e:
            logger.error(f"Error getting product matches: {e}")
            return []
    
    async def _track_impressions(
        self,
        suggestions: List[SearchSuggestion],
        session_id: str
    ) -> None:
        """Track suggestion impressions."""
        try:
            for suggestion in suggestions:
                if suggestion.id and not suggestion.id.startswith(("recent_", "popular_")):
                    # Update impression count
                    self.client.table("search_suggestions") \
                        .update({"impression_count": suggestion.impression_count + 1}) \
                        .eq("id", suggestion.id) \
                        .execute()
        except Exception as e:
            logger.warning(f"Error tracking impressions: {e}")

    async def check_typos(
        self,
        query: str,
        auto_apply_threshold: float = 0.9,
        max_suggestions: int = 3
    ) -> Tuple[bool, List[QueryCorrection], Optional[QueryCorrection]]:
        """
        Check query for typos and suggest corrections.

        Returns:
            Tuple of (has_corrections, corrections_list, recommended_correction)
        """
        corrections = []

        try:
            # 1. Check database for known corrections
            db_corrections = await self._get_known_corrections(query)
            corrections.extend(db_corrections)

            # 2. Check for abbreviation expansions
            abbrev_corrections = self._check_abbreviations(query)
            corrections.extend(abbrev_corrections)

            # 3. Check against popular searches for fuzzy matches
            fuzzy_corrections = await self._get_fuzzy_matches(query)
            corrections.extend(fuzzy_corrections)

            # Sort by confidence
            corrections.sort(key=lambda x: x.confidence_score, reverse=True)
            corrections = corrections[:max_suggestions]

            # Determine recommended correction
            recommended = None
            if corrections and corrections[0].confidence_score >= auto_apply_threshold:
                recommended = corrections[0]
                recommended.auto_applied = True

            return len(corrections) > 0, corrections, recommended

        except Exception as e:
            logger.error(f"Error checking typos: {e}")
            return False, [], None

    async def _get_known_corrections(self, query: str) -> List[QueryCorrection]:
        """Get known corrections from database."""
        try:
            response = self.client.table("search_query_corrections") \
                .select("*") \
                .eq("original_query", query.lower()) \
                .order("confidence_score", desc=True) \
                .limit(3) \
                .execute()

            if response.data:
                return [
                    QueryCorrection(
                        original_query=row["original_query"],
                        corrected_query=row["corrected_query"],
                        correction_type=row["correction_type"],
                        confidence_score=float(row["confidence_score"]),
                        auto_applied=False,
                        acceptance_rate=float(row.get("acceptance_rate", 0))
                    )
                    for row in response.data
                ]
            return []
        except Exception as e:
            logger.error(f"Error getting known corrections: {e}")
            return []

    def _check_abbreviations(self, query: str) -> List[QueryCorrection]:
        """Check for common abbreviations and expand them."""
        corrections = []
        words = query.lower().split()

        for i, word in enumerate(words):
            if word in self.abbreviations:
                expanded_words = words.copy()
                expanded_words[i] = self.abbreviations[word]
                corrected = " ".join(expanded_words)

                corrections.append(QueryCorrection(
                    original_query=query,
                    corrected_query=corrected,
                    correction_type="abbreviation",
                    confidence_score=0.85,
                    auto_applied=False,
                    acceptance_rate=0.7
                ))

        return corrections

    async def _get_fuzzy_matches(self, query: str) -> List[QueryCorrection]:
        """Get fuzzy matches from popular searches."""
        try:
            # Get popular searches
            response = self.client.table("search_analytics") \
                .select("query_text") \
                .order("created_at", desc=True) \
                .limit(1000) \
                .execute()

            if not response.data:
                return []

            # Find similar queries using fuzzy matching
            corrections = []

            for row in response.data:
                candidate = row["query_text"]

                # Skip exact matches
                if candidate.lower() == query.lower():
                    continue

                # Calculate similarity
                similarity = calculate_string_similarity(query, candidate, case_sensitive=False)

                # If similarity is high enough, suggest as correction
                if similarity >= 0.75:
                    corrections.append(QueryCorrection(
                        original_query=query,
                        corrected_query=row["query_text"],
                        correction_type="spelling",
                        confidence_score=similarity,
                        auto_applied=False,
                        acceptance_rate=0.6
                    ))

            # Sort by confidence and return top matches
            corrections.sort(key=lambda x: x.confidence_score, reverse=True)
            return corrections[:3]

        except Exception as e:
            logger.error(f"Error getting fuzzy matches: {e}")
            return []

    async def expand_query(
        self,
        query: str,
        max_synonyms_per_term: int = 3,
        max_related_concepts: int = 5,
        use_ai: bool = True
    ) -> ExpandedQuery:
        """
        Expand query with synonyms and related terms.
        """
        try:
            words = query.lower().split()
            expanded_terms = []
            synonyms = {}
            related_concepts = []

            # 1. Add synonyms for each word
            for word in words:
                if word in self.synonyms_map:
                    synonyms[word] = self.synonyms_map[word][:max_synonyms_per_term]
                    expanded_terms.extend(synonyms[word])

            # 2. Get related concepts from search analytics
            related = await self._get_related_concepts(query, max_related_concepts)
            related_concepts.extend(related)

            # 3. If AI is enabled, use Claude for semantic expansion
            if use_ai:
                try:
                    from app.services.core.ai_client_service import get_ai_client_service
                    ai_service = get_ai_client_service()

                    prompt = f"""Given the search query "{query}" for a materials database, suggest:
1. 5-10 semantically related terms
2. 3-5 broader concepts
3. 3-5 narrower/specific terms

Focus on materials, textures, colors, patterns, and applications.
Return as JSON: {{"related": [], "broader": [], "narrower": []}}"""

                    response = await ai_service.anthropic_async.messages.create(
                        model="claude-3-5-haiku-20241022",  # Fast, cheap model
                        max_tokens=500,
                        temperature=0.3,
                        messages=[{"role": "user", "content": prompt}]
                    )

                    import json
                    ai_suggestions = json.loads(response.content[0].text)

                    # Add AI suggestions to expanded terms
                    expanded_terms.extend(ai_suggestions.get("related", []))
                    expanded_terms.extend(ai_suggestions.get("narrower", []))
                    related_concepts.extend(ai_suggestions.get("broader", []))

                    logger.info(f"✅ AI semantic expansion added {len(ai_suggestions.get('related', []))} terms")

                except Exception as e:
                    logger.warning(f"AI semantic expansion failed, using fallback: {e}")

            return ExpandedQuery(
                original_query=query,
                expanded_terms=list(set(expanded_terms)),
                synonyms=synonyms,
                related_concepts=related_concepts,
                confidence_score=0.8
            )

        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return ExpandedQuery(
                original_query=query,
                expanded_terms=[],
                synonyms={},
                related_concepts=[],
                confidence_score=0.0
            )

    async def _get_related_concepts(self, query: str, limit: int) -> List[str]:
        """Get related concepts from search analytics."""
        try:
            # Get queries that users searched after this query
            response = self.client.table("search_analytics") \
                .select("follow_up_queries") \
                .ilike("query_text", f"%{query}%") \
                .not_.is_("follow_up_queries", "null") \
                .limit(100) \
                .execute()

            if not response.data:
                return []

            # Collect all follow-up queries
            related = []
            for row in response.data:
                if row.get("follow_up_queries"):
                    related.extend(row["follow_up_queries"])

            # Count frequency and return most common
            from collections import Counter
            counter = Counter(related)
            return [query for query, _ in counter.most_common(limit)]

        except Exception as e:
            logger.error(f"Error getting related concepts: {e}")
            return []

    async def track_suggestion_click(
        self,
        suggestion_id: str,
        user_id: Optional[str],
        session_id: Optional[str],
        original_query: str,
        suggestion_position: int,
        action_type: str,
        result_count: Optional[int] = None,
        user_satisfied: Optional[bool] = None
    ) -> bool:
        """Track when a user clicks on a suggestion."""
        try:
            # Insert click record
            self.client.table("search_suggestion_clicks").insert({
                "suggestion_id": suggestion_id,
                "user_id": user_id,
                "session_id": session_id,
                "original_query": original_query,
                "suggestion_position": suggestion_position,
                "action_type": action_type,
                "result_count": result_count,
                "user_satisfied": user_satisfied
            }).execute()

            return True
        except Exception as e:
            logger.error(f"Error tracking suggestion click: {e}")
            return False


