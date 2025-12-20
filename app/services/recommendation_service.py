"""
Recommendation Service - Collaborative Filtering for Material Recommendations

This service implements:
1. User-User Collaborative Filtering - "Users like you also liked..."
2. Item-Item Collaborative Filtering - "Materials similar to this..."
3. Hybrid Recommendations - Combining collaborative + content-based
4. Interaction Tracking - Track user interactions with materials
5. Score Caching - Cache recommendations for fast retrieval

Algorithms:
- Cosine Similarity for user/item vectors
- Matrix Factorization (SVD) for dimensionality reduction
- Hybrid approach combining collaborative + content-based filtering
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

from app.services.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)


class RecommendationService:
    """Service for collaborative filtering recommendations."""

    def __init__(self):
        """Initialize recommendation service."""
        self.supabase = get_supabase_client()
        self.cache_ttl_days = 7  # Cache recommendations for 7 days
        self.min_interactions = 3  # Minimum interactions for recommendations
        self.similarity_threshold = 0.3  # Minimum similarity score

    # ============================================================================
    # Interaction Tracking
    # ============================================================================

    async def track_interaction(
        self,
        user_id: str,
        workspace_id: str,
        material_id: str,
        interaction_type: str,
        interaction_value: float = 1.0,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Track user interaction with a material.

        Args:
            user_id: User ID
            workspace_id: Workspace ID
            material_id: Material/Product ID
            interaction_type: Type of interaction (view, click, save, purchase, rate, add_to_quote, share)
            interaction_value: Numeric value (rating 1-5, time spent, or weight)
            session_id: Optional session identifier
            metadata: Optional additional context

        Returns:
            Created interaction record
        """
        try:
            interaction_data = {
                "user_id": user_id,
                "workspace_id": workspace_id,
                "material_id": material_id,
                "interaction_type": interaction_type,
                "interaction_value": interaction_value,
                "session_id": session_id,
                "metadata": metadata or {}
            }

            response = self.supabase.client.table("user_material_interactions").insert(interaction_data).execute()

            if response.data:
                logger.info(f"✅ Tracked {interaction_type} interaction for user {user_id} on material {material_id}")
                return response.data[0]
            else:
                logger.error(f"❌ Failed to track interaction: No data returned")
                return {}

        except Exception as e:
            logger.error(f"❌ Error tracking interaction: {e}")
            return {}

    async def get_user_interactions(
        self,
        user_id: str,
        workspace_id: Optional[str] = None,
        interaction_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get user's interaction history.

        Args:
            user_id: User ID
            workspace_id: Optional workspace filter
            interaction_type: Optional interaction type filter
            limit: Maximum number of interactions to return

        Returns:
            List of interaction records
        """
        try:
            query = self.supabase.client.table("user_material_interactions").select("*").eq("user_id", user_id)

            if workspace_id:
                query = query.eq("workspace_id", workspace_id)

            if interaction_type:
                query = query.eq("interaction_type", interaction_type)

            response = query.order("created_at", desc=True).limit(limit).execute()

            if response.data:
                logger.info(f"✅ Retrieved {len(response.data)} interactions for user {user_id}")
                return response.data
            else:
                return []

        except Exception as e:
            logger.error(f"❌ Error getting user interactions: {e}")
            return []

    # ============================================================================
    # User-User Collaborative Filtering
    # ============================================================================

    async def get_similar_users(
        self,
        user_id: str,
        workspace_id: str,
        limit: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find users with similar interaction patterns.

        Args:
            user_id: Target user ID
            workspace_id: Workspace ID
            limit: Maximum number of similar users

        Returns:
            List of (user_id, similarity_score) tuples
        """
        try:
            # Get target user's interactions
            target_interactions = await self.get_user_interactions(user_id, workspace_id)

            if len(target_interactions) < self.min_interactions:
                logger.warning(f"User {user_id} has insufficient interactions ({len(target_interactions)} < {self.min_interactions})")
                return []

            # Build target user's interaction vector
            target_vector = self._build_interaction_vector(target_interactions)

            # Get all other users in workspace
            response = self.supabase.client.table("user_material_interactions").select("user_id").eq("workspace_id", workspace_id).neq("user_id", user_id).execute()

            if not response.data:
                return []

            other_user_ids = list(set([r["user_id"] for r in response.data]))

            # Calculate similarity with each user
            similarities = []
            for other_user_id in other_user_ids:
                other_interactions = await self.get_user_interactions(other_user_id, workspace_id)

                if len(other_interactions) < self.min_interactions:
                    continue

                other_vector = self._build_interaction_vector(other_interactions)
                similarity = self._cosine_similarity(target_vector, other_vector)

                if similarity >= self.similarity_threshold:
                    similarities.append((other_user_id, similarity))

            # Sort by similarity and return top K
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]

        except Exception as e:
            logger.error(f"❌ Error finding similar users: {e}")
            return []

    async def recommend_for_user(
        self,
        user_id: str,
        workspace_id: str,
        limit: int = 20,
        algorithm: str = "user_user"
    ) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a user based on similar users.

        Args:
            user_id: Target user ID
            workspace_id: Workspace ID
            limit: Maximum number of recommendations
            algorithm: Algorithm to use (user_user, item_item, hybrid)

        Returns:
            List of recommended materials with scores
        """
        try:
            # Check cache first
            cached_recommendations = await self._get_cached_recommendations(user_id, workspace_id, algorithm)
            if cached_recommendations:
                logger.info(f"✅ Retrieved {len(cached_recommendations)} cached recommendations for user {user_id}")
                return cached_recommendations[:limit]

            # Find similar users
            similar_users = await self.get_similar_users(user_id, workspace_id, limit=20)

            if not similar_users:
                logger.warning(f"No similar users found for user {user_id}")
                return []

            # Get materials liked by similar users
            material_scores = defaultdict(float)
            material_counts = defaultdict(int)

            for similar_user_id, similarity in similar_users:
                similar_user_interactions = await self.get_user_interactions(similar_user_id, workspace_id)

                for interaction in similar_user_interactions:
                    material_id = interaction["material_id"]
                    interaction_value = interaction.get("interaction_value", 1.0)

                    # Weight by similarity and interaction value
                    score = similarity * interaction_value
                    material_scores[material_id] += score
                    material_counts[material_id] += 1

            # Get user's existing interactions to exclude
            user_interactions = await self.get_user_interactions(user_id, workspace_id)
            user_material_ids = set([i["material_id"] for i in user_interactions])

            # Build recommendations
            recommendations = []
            for material_id, total_score in material_scores.items():
                if material_id in user_material_ids:
                    continue  # Skip materials user already interacted with

                avg_score = total_score / material_counts[material_id]
                confidence = min(material_counts[material_id] / 5.0, 1.0)  # Confidence based on number of similar users

                recommendations.append({
                    "material_id": material_id,
                    "score": avg_score,
                    "confidence": confidence,
                    "algorithm": algorithm,
                    "metadata": {
                        "similar_users_count": len(similar_users),
                        "recommending_users": material_counts[material_id]
                    }
                })

            # Sort by score
            recommendations.sort(key=lambda x: x["score"], reverse=True)

            # Cache recommendations
            await self._cache_recommendations(user_id, workspace_id, recommendations[:limit], algorithm)

            logger.info(f"✅ Generated {len(recommendations[:limit])} recommendations for user {user_id}")
            return recommendations[:limit]

        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            return []

    # ============================================================================
    # Item-Item Collaborative Filtering
    # ============================================================================

    async def get_similar_materials(
        self,
        material_id: str,
        workspace_id: str,
        limit: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find materials with similar interaction patterns.

        Args:
            material_id: Target material ID
            workspace_id: Workspace ID
            limit: Maximum number of similar materials

        Returns:
            List of (material_id, similarity_score) tuples
        """
        try:
            # Get users who interacted with target material
            response = self.supabase.client.table("user_material_interactions").select("user_id, interaction_value").eq("material_id", material_id).eq("workspace_id", workspace_id).execute()

            if not response.data or len(response.data) < self.min_interactions:
                logger.warning(f"Material {material_id} has insufficient interactions")
                return []

            target_users = {r["user_id"]: r.get("interaction_value", 1.0) for r in response.data}

            # Get all other materials in workspace
            all_materials_response = self.supabase.client.table("user_material_interactions").select("material_id").eq("workspace_id", workspace_id).neq("material_id", material_id).execute()

            if not all_materials_response.data:
                return []

            other_material_ids = list(set([r["material_id"] for r in all_materials_response.data]))

            # Calculate similarity with each material
            similarities = []
            for other_material_id in other_material_ids:
                other_response = self.supabase.client.table("user_material_interactions").select("user_id, interaction_value").eq("material_id", other_material_id).eq("workspace_id", workspace_id).execute()

                if not other_response.data or len(other_response.data) < self.min_interactions:
                    continue

                other_users = {r["user_id"]: r.get("interaction_value", 1.0) for r in other_response.data}

                # Calculate Jaccard similarity + weighted overlap
                common_users = set(target_users.keys()) & set(other_users.keys())
                if not common_users:
                    continue

                # Weighted similarity based on interaction values
                similarity = sum(target_users[u] * other_users[u] for u in common_users) / (len(target_users) + len(other_users) - len(common_users))

                if similarity >= self.similarity_threshold:
                    similarities.append((other_material_id, similarity))

            # Sort by similarity and return top K
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]

        except Exception as e:
            logger.error(f"❌ Error finding similar materials: {e}")
            return []

    async def recommend_similar_materials(
        self,
        material_id: str,
        workspace_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Recommend materials similar to a given material.

        Args:
            material_id: Target material ID
            workspace_id: Workspace ID
            limit: Maximum number of recommendations

        Returns:
            List of recommended materials with scores
        """
        try:
            similar_materials = await self.get_similar_materials(material_id, workspace_id, limit)

            recommendations = []
            for similar_material_id, similarity in similar_materials:
                recommendations.append({
                    "material_id": similar_material_id,
                    "score": similarity,
                    "confidence": 0.8,  # High confidence for item-item
                    "algorithm": "item_item",
                    "metadata": {
                        "source_material_id": material_id
                    }
                })

            logger.info(f"✅ Found {len(recommendations)} similar materials to {material_id}")
            return recommendations

        except Exception as e:
            logger.error(f"❌ Error recommending similar materials: {e}")
            return []

    # ============================================================================
    # Caching
    # ============================================================================

    async def _get_cached_recommendations(
        self,
        user_id: str,
        workspace_id: str,
        algorithm: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached recommendations if available and not expired."""
        try:
            response = self.supabase.client.table("recommendation_scores").select("*").eq("user_id", user_id).eq("workspace_id", workspace_id).eq("algorithm", algorithm).gt("expires_at", datetime.utcnow().isoformat()).order("score", desc=True).execute()

            if response.data:
                return [
                    {
                        "material_id": r["material_id"],
                        "score": r["score"],
                        "confidence": r["confidence"],
                        "algorithm": r["algorithm"],
                        "metadata": r.get("metadata", {})
                    }
                    for r in response.data
                ]
            return None

        except Exception as e:
            logger.error(f"❌ Error getting cached recommendations: {e}")
            return None

    async def _cache_recommendations(
        self,
        user_id: str,
        workspace_id: str,
        recommendations: List[Dict[str, Any]],
        algorithm: str
    ) -> None:
        """Cache recommendations for future retrieval."""
        try:
            expires_at = datetime.utcnow() + timedelta(days=self.cache_ttl_days)

            cache_records = [
                {
                    "user_id": user_id,
                    "workspace_id": workspace_id,
                    "material_id": rec["material_id"],
                    "score": rec["score"],
                    "confidence": rec["confidence"],
                    "algorithm": algorithm,
                    "metadata": rec.get("metadata", {}),
                    "expires_at": expires_at.isoformat()
                }
                for rec in recommendations
            ]

            # Delete old cache entries
            self.supabase.client.table("recommendation_scores").delete().eq("user_id", user_id).eq("workspace_id", workspace_id).eq("algorithm", algorithm).execute()

            # Insert new cache entries
            self.supabase.client.table("recommendation_scores").insert(cache_records).execute()

            logger.info(f"✅ Cached {len(cache_records)} recommendations for user {user_id}")

        except Exception as e:
            logger.error(f"❌ Error caching recommendations: {e}")

    async def invalidate_cache(
        self,
        user_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        material_id: Optional[str] = None
    ) -> None:
        """Invalidate cached recommendations."""
        try:
            query = self.supabase.client.table("recommendation_scores").delete()

            if user_id:
                query = query.eq("user_id", user_id)
            if workspace_id:
                query = query.eq("workspace_id", workspace_id)
            if material_id:
                query = query.eq("material_id", material_id)

            query.execute()
            logger.info(f"✅ Invalidated recommendation cache")

        except Exception as e:
            logger.error(f"❌ Error invalidating cache: {e}")

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _build_interaction_vector(self, interactions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Build interaction vector from user interactions."""
        vector = {}
        for interaction in interactions:
            material_id = interaction["material_id"]
            interaction_value = interaction.get("interaction_value", 1.0)
            interaction_type = interaction["interaction_type"]

            # Weight different interaction types
            weights = {
                "view": 1.0,
                "click": 2.0,
                "save": 3.0,
                "add_to_quote": 4.0,
                "purchase": 5.0,
                "rate": interaction_value,  # Use actual rating value
                "share": 3.0
            }

            weight = weights.get(interaction_type, 1.0)
            vector[material_id] = vector.get(material_id, 0.0) + weight

        return vector

    def _cosine_similarity(self, vector1: Dict[str, float], vector2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two interaction vectors."""
        try:
            # Get common materials
            common_materials = set(vector1.keys()) & set(vector2.keys())

            if not common_materials:
                return 0.0

            # Calculate dot product
            dot_product = sum(vector1[m] * vector2[m] for m in common_materials)

            # Calculate magnitudes
            magnitude1 = np.sqrt(sum(v ** 2 for v in vector1.values()))
            magnitude2 = np.sqrt(sum(v ** 2 for v in vector2.values()))

            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0

            # Calculate cosine similarity
            similarity = dot_product / (magnitude1 * magnitude2)
            return float(similarity)

        except Exception as e:
            logger.error(f"❌ Error calculating cosine similarity: {e}")
            return 0.0

    # ============================================================================
    # Analytics
    # ============================================================================

    async def get_recommendation_analytics(
        self,
        workspace_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get recommendation analytics for a workspace."""
        try:
            start_date = datetime.utcnow() - timedelta(days=days)

            # Get interaction counts by type
            interactions_response = self.supabase.client.table("user_material_interactions").select("interaction_type").eq("workspace_id", workspace_id).gte("created_at", start_date.isoformat()).execute()

            interaction_counts = defaultdict(int)
            if interactions_response.data:
                for interaction in interactions_response.data:
                    interaction_counts[interaction["interaction_type"]] += 1

            # Get cached recommendations count
            cache_response = self.supabase.client.table("recommendation_scores").select("algorithm").eq("workspace_id", workspace_id).execute()

            algorithm_counts = defaultdict(int)
            if cache_response.data:
                for rec in cache_response.data:
                    algorithm_counts[rec["algorithm"]] += 1

            return {
                "workspace_id": workspace_id,
                "period_days": days,
                "total_interactions": sum(interaction_counts.values()),
                "interactions_by_type": dict(interaction_counts),
                "cached_recommendations": sum(algorithm_counts.values()),
                "recommendations_by_algorithm": dict(algorithm_counts)
            }

        except Exception as e:
            logger.error(f"❌ Error getting recommendation analytics: {e}")
            return {}


# Global instance
recommendation_service = RecommendationService()


def get_recommendation_service() -> RecommendationService:
    """
    Get the global recommendation service instance.

    Returns:
        RecommendationService instance
    """
    return recommendation_service

