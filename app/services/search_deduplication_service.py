"""
Search Deduplication Service

Implements AI-powered smart deduplication for saved searches to prevent
database bloat while respecting important contextual differences.

Features:
- Claude Haiku 4.5 for semantic analysis
- CLIP embeddings for similarity matching
- Multi-layer matching (exact, semantic, metadata)
- Context-aware merging (floor vs wall, indoor vs outdoor)
- Attribute conflict detection
"""

import json
import asyncio
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from anthropic import AsyncAnthropic
import openai

from ..config import get_settings
from .supabase_client import get_supabase_client

# Get settings
settings = get_settings()


@dataclass
class SearchAnalysis:
    """Structured analysis of a search query."""
    core_material: str
    attributes: Dict[str, Any]
    application_context: Optional[str]
    intent_category: str
    semantic_fingerprint: List[float]
    normalized_query: str


class SearchDeduplicationService:
    """Service for intelligent search deduplication."""
    
    def __init__(self):
        self.anthropic = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.openai = openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.supabase = get_supabase_client()
        
        # Configuration
        self.SEMANTIC_THRESHOLD = 0.85
        self.AUTO_MERGE_THRESHOLD = 0.95
        self.REQUIRE_CONTEXT_MATCH = True
        self.ALLOW_NULL_CONTEXT_MERGE = True
    
    async def analyze_search_query(self, query: str) -> SearchAnalysis:
        """
        Use Claude Haiku 4.5 to extract semantic components from search query.
        
        Args:
            query: User's search query
            
        Returns:
            SearchAnalysis with extracted components
        """
        
        prompt = f"""Analyze this material search query and extract structured information:

Query: "{query}"

Extract:
1. Core Material: The main material/product (e.g., "cement tile", "oak flooring", "marble countertop")
2. Attributes: Specific properties as key-value pairs
   - color: if mentioned (e.g., "grey", "white", "natural")
   - texture: if mentioned (e.g., "smooth", "rough", "matte", "glossy")
   - finish: if mentioned (e.g., "polished", "honed", "brushed")
   - size: if mentioned (e.g., "large", "small", "60x60cm")
   - any other specific attributes
3. Application Context: Where it will be used (e.g., "floor", "wall", "outdoor", "indoor", "kitchen", "bathroom")
   - Be specific: "kitchen floor" not just "kitchen"
   - If outdoor/indoor mentioned, include it
4. Intent Category: One of: product_search, comparison, recommendation, specification

Return ONLY valid JSON, no markdown:
{{
  "core_material": "...",
  "attributes": {{}},
  "application_context": "...",
  "intent_category": "..."
}}

Rules:
- If attribute not explicitly mentioned, omit it from attributes object
- If no context mentioned, use null for application_context
- Be precise and consistent with naming
- Normalize colors (gray→grey, etc)"""

        try:
            response = await self.anthropic.messages.create(
                model="claude-haiku-4.5",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse AI response
            content = response.content[0].text.strip()
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            analysis = json.loads(content)
            
            # Generate CLIP embedding for semantic similarity
            embedding = await self._generate_clip_embedding(query)
            
            # Normalize query
            normalized = self._normalize_query(query)
            
            return SearchAnalysis(
                core_material=analysis.get("core_material", "").lower(),
                attributes=analysis.get("attributes", {}),
                application_context=analysis.get("application_context"),
                intent_category=analysis.get("intent_category", "product_search"),
                semantic_fingerprint=embedding,
                normalized_query=normalized
            )
            
        except Exception as e:
            print(f"Error analyzing query: {e}")
            # Fallback to simple analysis
            return SearchAnalysis(
                core_material=query.lower(),
                attributes={},
                application_context=None,
                intent_category="product_search",
                semantic_fingerprint=await self._generate_clip_embedding(query),
                normalized_query=self._normalize_query(query)
            )
    
    async def _generate_clip_embedding(self, text: str) -> List[float]:
        """Generate CLIP embedding for text."""
        try:
            response = await self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for exact matching."""
        # Convert to lowercase
        normalized = query.lower()
        
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        
        # Normalize common variations
        replacements = {
            "gray": "grey",
            "flooring": "floor",
            "tiles": "tile",
            "i need": "",
            "looking for": "",
            "searching for": "",
            "i want": "",
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized.strip()
    
    async def find_or_merge_search(
        self,
        user_id: str,
        query: str,
        filters: Dict[str, Any],
        material_filters: Dict[str, Any]
    ) -> Tuple[Optional[str], bool, Optional[Dict]]:
        """
        Find existing similar search or determine if new one should be created.
        
        Args:
            user_id: User ID
            query: Search query
            filters: General filters
            material_filters: Material-specific filters
            
        Returns:
            Tuple of (existing_search_id, should_merge, merge_suggestion)
            - If should_merge=True and existing_search_id: auto-merge
            - If should_merge=False and merge_suggestion: show user choice
            - If both None: create new search
        """
        
        # 1. Analyze query with AI
        analysis = await self.analyze_search_query(query)
        
        # 2. Find similar searches
        similar_searches = await self._find_similar_searches(
            user_id=user_id,
            analysis=analysis
        )
        
        if not similar_searches:
            return (None, False, None)
        
        # 3. Check each similar search for merge eligibility
        for existing in similar_searches:
            similarity_score = existing.get("similarity_score", 0)
            
            # Check if should merge
            can_merge, reason = self._should_merge(
                existing=existing,
                new_analysis=analysis,
                new_filters=material_filters
            )
            
            if can_merge:
                # Auto-merge if very high similarity
                if similarity_score >= self.AUTO_MERGE_THRESHOLD:
                    return (existing["id"], True, None)
                
                # Suggest merge to user
                merge_suggestion = {
                    "existing_search": existing,
                    "similarity_score": similarity_score,
                    "reason": reason,
                    "new_query": query
                }
                return (existing["id"], False, merge_suggestion)
        
        # No suitable merge found
        return (None, False, None)
    
    async def _find_similar_searches(
        self,
        user_id: str,
        analysis: SearchAnalysis
    ) -> List[Dict]:
        """Find similar saved searches using multi-layer matching."""
        
        # Query database for potential matches
        # Layer 1: Same core material
        # Layer 2: Semantic similarity via embedding
        # Layer 3: Context matching
        
        try:
            # Get searches with same core material
            response = self.supabase.from_("saved_searches").select("*").eq(
                "user_id", user_id
            ).eq(
                "core_material", analysis.core_material
            ).limit(20).execute()
            
            candidates = response.data if response.data else []
            
            # Calculate similarity scores
            results = []
            for candidate in candidates:
                if not candidate.get("semantic_fingerprint"):
                    continue
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(
                    analysis.semantic_fingerprint,
                    candidate["semantic_fingerprint"]
                )
                
                if similarity >= self.SEMANTIC_THRESHOLD:
                    candidate["similarity_score"] = similarity
                    results.append(candidate)
            
            # Sort by similarity
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            
            return results[:10]  # Top 10 matches

        except Exception as e:
            print(f"Error finding similar searches: {e}")
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            v1 = np.array(vec1)
            v2 = np.array(vec2)

            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def _should_merge(
        self,
        existing: Dict,
        new_analysis: SearchAnalysis,
        new_filters: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Determine if new search should merge with existing.

        Returns:
            Tuple of (should_merge, reason)
        """

        # 1. Core material must match (already filtered)

        # 2. Application context must match (critical!)
        existing_context = existing.get("application_context")
        new_context = new_analysis.application_context

        if self.REQUIRE_CONTEXT_MATCH:
            if existing_context != new_context:
                # Exception: if both are null/generic, allow merge
                if not (self.ALLOW_NULL_CONTEXT_MERGE and
                       not existing_context and not new_context):
                    return (False, "Different application context")

        # 3. Check for conflicting attributes
        existing_attrs = existing.get("material_attributes", {})
        new_attrs = new_analysis.attributes

        has_conflict, conflict_reason = self._has_conflicting_attributes(
            existing_attrs,
            new_attrs
        )

        if has_conflict:
            return (False, conflict_reason)

        # 4. Check filter compatibility
        existing_filters = existing.get("material_filters", {})

        if not self._filters_compatible(existing_filters, new_filters):
            return (False, "Incompatible filters")

        # All checks passed
        return (True, "Compatible search - can merge")

    def _has_conflicting_attributes(
        self,
        existing_attrs: Dict,
        new_attrs: Dict
    ) -> Tuple[bool, str]:
        """
        Check if attributes conflict.

        Examples:
        - {color: grey} + {color: white} → CONFLICT
        - {color: grey} + {texture: smooth} → NO CONFLICT (additive)
        - {outdoor: true} + {indoor: true} → CONFLICT
        """

        # Define mutually exclusive attribute pairs
        conflicts = [
            ("outdoor", "indoor"),
            ("wall", "floor"),
            ("matte", "glossy"),
            ("polished", "honed"),
        ]

        # Check for same-key conflicts (different values)
        for key in existing_attrs:
            if key in new_attrs:
                if existing_attrs[key] != new_attrs[key]:
                    return (True, f"Conflicting {key}: {existing_attrs[key]} vs {new_attrs[key]}")

        # Check for mutually exclusive attributes
        for key1, key2 in conflicts:
            if key1 in existing_attrs and key2 in new_attrs:
                return (True, f"Conflicting attributes: {key1} vs {key2}")
            if key2 in existing_attrs and key1 in new_attrs:
                return (True, f"Conflicting attributes: {key2} vs {key1}")

        return (False, "")

    def _filters_compatible(
        self,
        existing_filters: Dict,
        new_filters: Dict
    ) -> bool:
        """Check if material filters are compatible for merging."""

        # For now, allow merge if filters are similar or one is subset of other
        # This can be made more sophisticated based on requirements

        # If either is empty, compatible
        if not existing_filters or not new_filters:
            return True

        # Check critical filter differences
        # Colors must match if both specified
        existing_colors = existing_filters.get("colors", [])
        new_colors = new_filters.get("colors", [])

        if existing_colors and new_colors:
            # Must have some overlap
            if not set(existing_colors) & set(new_colors):
                return False

        # Price ranges must overlap if both specified
        existing_price = existing_filters.get("priceRange", [0, 10000])
        new_price = new_filters.get("priceRange", [0, 10000])

        if existing_price and new_price:
            # Check for overlap
            if existing_price[1] < new_price[0] or new_price[1] < existing_price[0]:
                return False

        return True

    async def merge_into_existing(
        self,
        existing_id: str,
        new_query: str,
        new_filters: Dict,
        new_material_filters: Dict,
        analysis: SearchAnalysis
    ) -> str:
        """
        Merge new search into existing one.

        Strategy:
        1. Keep most specific query as primary
        2. Merge attributes (union, no conflicts)
        3. Update filters to be more inclusive
        4. Increment merge_count
        5. Update last_merged_at
        """

        try:
            # Get existing search
            response = self.supabase.from_("saved_searches").select("*").eq(
                "id", existing_id
            ).single().execute()

            existing = response.data

            # Choose better query (more specific wins)
            updated_query = self._choose_better_query(
                existing["query"],
                new_query
            )

            # Merge attributes (union)
            merged_attributes = {
                **existing.get("material_attributes", {}),
                **analysis.attributes
            }

            # Merge filters (more inclusive)
            merged_filters = self._merge_filters(
                existing.get("material_filters", {}),
                new_material_filters
            )

            # Update search
            update_data = {
                "query": updated_query,
                "material_attributes": merged_attributes,
                "material_filters": merged_filters,
                "merge_count": existing.get("merge_count", 1) + 1,
                "last_merged_at": "now()",
                "updated_at": "now()"
            }

            self.supabase.from_("saved_searches").update(update_data).eq(
                "id", existing_id
            ).execute()

            return existing_id

        except Exception as e:
            print(f"Error merging search: {e}")
            raise

    def _choose_better_query(self, existing: str, new: str) -> str:
        """Choose more specific/descriptive query."""

        # Simple heuristic: longer with more detail wins
        if len(new.split()) > len(existing.split()):
            return new
        return existing

    def _merge_filters(
        self,
        existing_filters: Dict,
        new_filters: Dict
    ) -> Dict:
        """Merge filters to be more inclusive."""

        merged = {**existing_filters}

        # Merge arrays (union)
        for key in ["materialTypes", "colors", "suppliers", "applications", "textures"]:
            if key in new_filters:
                existing_values = set(merged.get(key, []))
                new_values = set(new_filters[key])
                merged[key] = list(existing_values | new_values)

        # Merge price range (expand to include both)
        if "priceRange" in new_filters:
            existing_range = merged.get("priceRange", [0, 10000])
            new_range = new_filters["priceRange"]
            merged["priceRange"] = [
                min(existing_range[0], new_range[0]),
                max(existing_range[1], new_range[1])
            ]

        return merged


# Singleton instance
_deduplication_service = None

def get_deduplication_service() -> SearchDeduplicationService:
    """Get singleton instance of deduplication service."""
    global _deduplication_service
    if _deduplication_service is None:
        _deduplication_service = SearchDeduplicationService()
    return _deduplication_service

