"""
Product Validation Service

Validates extracted products against quality thresholds:
- Minimum chunks/characters
- Substantive content check
- Distinguishing features validation
- Associated assets verification
- Semantic coherence scoring

Prevents false positives and ensures high-quality product extraction.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import numpy as np

from app.services.real_embeddings_service import RealEmbeddingsService

logger = logging.getLogger(__name__)


class ProductValidator:
    """
    Validates extracted products against quality criteria.
    
    Validation checks:
    1. Minimum content requirements
    2. Substantive content (not just headers/footers)
    3. Distinguishing features present
    4. Associated assets (images, specs)
    5. Semantic coherence
    """
    
    # Validation thresholds
    MIN_CHUNKS = 2              # Minimum chunks per product
    MIN_CHARACTERS = 200        # Minimum total characters
    MIN_SUBSTANTIVE_RATIO = 0.6 # Minimum ratio of substantive content
    MIN_COHERENCE_SCORE = 0.65  # Minimum semantic coherence
    MIN_OVERALL_SCORE = 0.70    # Minimum overall validation score
    
    # Distinguishing features (at least one required)
    DISTINGUISHING_FEATURES = [
        "product name",
        "model number",
        "specifications",
        "dimensions",
        "materials",
        "features",
        "price",
        "sku",
        "part number",
    ]
    
    # Non-substantive content patterns
    NON_SUBSTANTIVE_PATTERNS = [
        "page",
        "copyright",
        "all rights reserved",
        "table of contents",
        "index",
        "www.",
        "http",
        "email",
        "phone",
    ]
    
    def __init__(self):
        """Initialize product validator."""
        self.embeddings_service = RealEmbeddingsService()
    
    async def validate_product(
        self,
        product_data: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        images: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Validate a product extraction.
        
        Args:
            product_data: Product metadata
            chunks: List of chunks associated with product
            images: Optional list of associated images
            
        Returns:
            Validation result with score, passed, and details
        """
        validation_results = {}
        
        # 1. Check minimum content requirements
        content_check = self._check_minimum_content(chunks)
        validation_results["content_requirements"] = content_check
        
        # 2. Check substantive content
        substantive_check = self._check_substantive_content(chunks)
        validation_results["substantive_content"] = substantive_check
        
        # 3. Check distinguishing features
        features_check = self._check_distinguishing_features(product_data, chunks)
        validation_results["distinguishing_features"] = features_check
        
        # 4. Check associated assets
        assets_check = self._check_associated_assets(images, chunks)
        validation_results["associated_assets"] = assets_check
        
        # 5. Check semantic coherence
        coherence_check = await self._check_semantic_coherence(chunks)
        validation_results["semantic_coherence"] = coherence_check
        
        # Calculate overall score (weighted average)
        overall_score = (
            content_check["score"] * 0.20 +
            substantive_check["score"] * 0.20 +
            features_check["score"] * 0.25 +
            assets_check["score"] * 0.15 +
            coherence_check["score"] * 0.20
        )
        
        passed = overall_score >= self.MIN_OVERALL_SCORE
        
        logger.info(
            f"{'✅' if passed else '❌'} Product validation: "
            f"score={overall_score:.2f}, passed={passed}"
        )
        
        return {
            "passed": passed,
            "overall_score": overall_score,
            "details": validation_results,
            "threshold": self.MIN_OVERALL_SCORE,
        }
    
    def _check_minimum_content(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check minimum content requirements.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Check result with score and details
        """
        chunk_count = len(chunks)
        total_chars = sum(len(chunk.get("content", "")) for chunk in chunks)
        
        chunk_score = min(1.0, chunk_count / self.MIN_CHUNKS)
        char_score = min(1.0, total_chars / self.MIN_CHARACTERS)
        
        score = (chunk_score + char_score) / 2
        passed = chunk_count >= self.MIN_CHUNKS and total_chars >= self.MIN_CHARACTERS
        
        return {
            "passed": passed,
            "score": score,
            "chunk_count": chunk_count,
            "total_characters": total_chars,
            "min_chunks": self.MIN_CHUNKS,
            "min_characters": self.MIN_CHARACTERS,
        }
    
    def _check_substantive_content(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check for substantive content (not just headers/footers).
        
        Args:
            chunks: List of chunks
            
        Returns:
            Check result with score and details
        """
        total_chars = 0
        substantive_chars = 0
        
        for chunk in chunks:
            content = chunk.get("content", "")
            total_chars += len(content)
            
            # Check if content is substantive
            content_lower = content.lower()
            is_non_substantive = any(
                pattern in content_lower
                for pattern in self.NON_SUBSTANTIVE_PATTERNS
            )
            
            if not is_non_substantive:
                substantive_chars += len(content)
        
        if total_chars == 0:
            ratio = 0.0
        else:
            ratio = substantive_chars / total_chars
        
        score = min(1.0, ratio / self.MIN_SUBSTANTIVE_RATIO)
        passed = ratio >= self.MIN_SUBSTANTIVE_RATIO
        
        return {
            "passed": passed,
            "score": score,
            "substantive_ratio": ratio,
            "min_ratio": self.MIN_SUBSTANTIVE_RATIO,
        }
    
    def _check_distinguishing_features(
        self,
        product_data: Dict[str, Any],
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Check for distinguishing features.
        
        Args:
            product_data: Product metadata
            chunks: List of chunks
            
        Returns:
            Check result with score and details
        """
        # Combine all content
        all_content = " ".join(chunk.get("content", "") for chunk in chunks)
        all_content += " " + str(product_data)
        all_content_lower = all_content.lower()
        
        # Check for features
        features_found = []
        for feature in self.DISTINGUISHING_FEATURES:
            if feature.lower() in all_content_lower:
                features_found.append(feature)
        
        feature_count = len(features_found)
        score = min(1.0, feature_count / 3)  # Need at least 3 features for perfect score
        passed = feature_count >= 1  # At least one feature required
        
        return {
            "passed": passed,
            "score": score,
            "features_found": features_found,
            "feature_count": feature_count,
        }
    
    def _check_associated_assets(
        self,
        images: Optional[List[Dict[str, Any]]],
        chunks: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Check for associated assets (images, specs).
        
        Args:
            images: Optional list of images
            chunks: List of chunks
            
        Returns:
            Check result with score and details
        """
        image_count = len(images) if images else 0
        
        # Check for specification content
        has_specs = any(
            "specification" in chunk.get("content", "").lower() or
            "spec" in chunk.get("content", "").lower()
            for chunk in chunks
        )
        
        # Score based on assets
        asset_score = 0.0
        
        if image_count > 0:
            asset_score += 0.5
        
        if image_count >= 2:
            asset_score += 0.2
        
        if has_specs:
            asset_score += 0.3
        
        score = min(1.0, asset_score)
        passed = image_count > 0 or has_specs
        
        return {
            "passed": passed,
            "score": score,
            "image_count": image_count,
            "has_specifications": has_specs,
        }
    
    async def _check_semantic_coherence(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check semantic coherence of chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Check result with score and details
        """
        if len(chunks) < 2:
            # Single chunk is always coherent
            return {
                "passed": True,
                "score": 1.0,
                "coherence_score": 1.0,
            }
        
        try:
            # Get embeddings for all chunks
            embeddings = []
            for chunk in chunks:
                result = await self.embeddings_service.generate_embedding(
                    text=chunk.get("content", ""),
                    embedding_type="text",
                )
                
                if result.get("success") and result.get("embedding"):
                    embeddings.append(np.array(result["embedding"]))
            
            if len(embeddings) < 2:
                # Not enough embeddings
                return {
                    "passed": False,
                    "score": 0.5,
                    "coherence_score": 0.5,
                    "error": "Failed to generate embeddings",
                }
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(embeddings) - 1):
                for j in range(i + 1, len(embeddings)):
                    sim = self._cosine_similarity(embeddings[i], embeddings[j])
                    similarities.append(sim)
            
            # Average similarity = coherence score
            coherence_score = np.mean(similarities) if similarities else 0.5
            
            score = min(1.0, coherence_score / self.MIN_COHERENCE_SCORE)
            passed = coherence_score >= self.MIN_COHERENCE_SCORE
            
            return {
                "passed": passed,
                "score": score,
                "coherence_score": float(coherence_score),
                "min_coherence": self.MIN_COHERENCE_SCORE,
            }
            
        except Exception as e:
            logger.error(f"❌ Coherence check failed: {str(e)}")
            return {
                "passed": False,
                "score": 0.5,
                "coherence_score": 0.5,
                "error": str(e),
            }
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float((similarity + 1) / 2)  # Normalize to 0-1
        except Exception:
            return 0.5


