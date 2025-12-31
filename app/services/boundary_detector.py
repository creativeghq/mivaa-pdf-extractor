"""
Product Boundary Detection Service

Detects product boundaries in PDF documents using:
- Semantic embeddings
- Structural markers (page breaks, headers, images)
- Content similarity analysis
- Visual layout analysis

Helps identify where one product ends and another begins.
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging
import numpy as np

from app.services.real_embeddings_service import RealEmbeddingsService
from app.services.ai_call_logger import AICallLogger

logger = logging.getLogger(__name__)


class BoundaryDetector:
    """
    Detects product boundaries in document content.
    
    Uses multiple signals:
    - Semantic similarity between chunks
    - Structural markers (page breaks, headers)
    - Visual layout changes
    - Content type transitions
    """
    
    # Similarity threshold for boundary detection
    SIMILARITY_THRESHOLD = 0.65  # Below this = likely boundary
    
    # Structural markers that indicate boundaries
    BOUNDARY_MARKERS = [
        "new product",
        "product name:",
        "model:",
        "collection:",
        "series:",
    ]
    
    def __init__(self, ai_logger: Optional[AICallLogger] = None):
        """
        Initialize boundary detector.
        
        Args:
            ai_logger: AI call logger instance
        """
        self.embeddings_service = RealEmbeddingsService()
        self.ai_logger = ai_logger or AICallLogger()
    
    async def detect_boundaries(
        self,
        chunks: List[Dict[str, Any]],
        job_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect product boundaries in a list of chunks.
        
        Args:
            chunks: List of document chunks with content and metadata
            job_id: Optional job ID for logging
            
        Returns:
            List of boundary detections with indices and confidence
        """
        if len(chunks) < 2:
            return []
        
        boundaries = []
        
        # Get embeddings for all chunks
        chunk_texts = [chunk.get("content", "") for chunk in chunks]
        embeddings = await self._get_embeddings_batch(chunk_texts, job_id)
        
        # Analyze consecutive chunks for boundaries
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Calculate similarity
            if embeddings[i] is not None and embeddings[i + 1] is not None:
                similarity = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            else:
                similarity = 0.5  # Default if embeddings failed
            
            # Check for structural markers
            has_marker = self._has_boundary_marker(next_chunk.get("content", ""))
            
            # Check for page break
            current_page = current_chunk.get("metadata", {}).get("page_number", 0)
            next_page = next_chunk.get("metadata", {}).get("page_number", 0)
            page_break = next_page > current_page
            
            # Check for image presence (products often have images)
            has_image = next_chunk.get("metadata", {}).get("has_images", False)
            
            # Calculate boundary confidence
            confidence = self._calculate_boundary_confidence(
                similarity=similarity,
                has_marker=has_marker,
                page_break=page_break,
                has_image=has_image,
            )
            
            # If confidence is high enough, mark as boundary
            if confidence >= 0.6:
                boundaries.append({
                    "index": i + 1,  # Boundary is BEFORE this chunk
                    "confidence": confidence,
                    "similarity": similarity,
                    "has_marker": has_marker,
                    "page_break": page_break,
                    "has_image": has_image,
                    "reason": self._get_boundary_reason(
                        similarity, has_marker, page_break, has_image
                    ),
                })
                
                logger.info(
                    f"🔍 Boundary detected at chunk {i + 1} "
                    f"(confidence: {confidence:.2f}, similarity: {similarity:.2f})"
                )
        
        return boundaries
    
    async def _get_embeddings_batch(
        self,
        texts: List[str],
        job_id: Optional[str],
    ) -> List[Optional[np.ndarray]]:
        """
        Get embeddings for a batch of texts.
        
        Args:
            texts: List of text strings
            job_id: Optional job ID
            
        Returns:
            List of embedding arrays (or None if failed)
        """
        embeddings = []
        
        for text in texts:
            try:
                result = await self.embeddings_service.generate_embedding(
                    text=text,
                    embedding_type="text",
                )
                
                if result.get("success") and result.get("embedding"):
                    embeddings.append(np.array(result["embedding"]))
                else:
                    embeddings.append(None)
            except Exception as e:
                logger.error(f"❌ Failed to get embedding: {str(e)}")
                embeddings.append(None)
        
        return embeddings
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Similarity score (0.0-1.0)
        """
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Normalize to 0-1 range
            return float((similarity + 1) / 2)
        except Exception as e:
            logger.error(f"❌ Similarity calculation failed: {str(e)}")
            return 0.5
    
    def _has_boundary_marker(self, content: str) -> bool:
        """
        Check if content has boundary markers.
        
        Args:
            content: Text content
            
        Returns:
            True if has markers
        """
        content_lower = content.lower()
        return any(marker in content_lower for marker in self.BOUNDARY_MARKERS)
    
    def _calculate_boundary_confidence(
        self,
        similarity: float,
        has_marker: bool,
        page_break: bool,
        has_image: bool,
    ) -> float:
        """
        Calculate boundary confidence based on multiple signals.
        
        Args:
            similarity: Semantic similarity (0.0-1.0)
            has_marker: Has structural marker
            page_break: Has page break
            has_image: Has image
            
        Returns:
            Confidence score (0.0-1.0)
        """
        # Start with similarity-based confidence
        # Low similarity = high boundary confidence
        similarity_confidence = 1.0 - similarity
        
        # Boost confidence for markers
        if has_marker:
            similarity_confidence = min(1.0, similarity_confidence + 0.3)
        
        # Boost for page breaks
        if page_break:
            similarity_confidence = min(1.0, similarity_confidence + 0.15)
        
        # Boost for images (products often have images)
        if has_image:
            similarity_confidence = min(1.0, similarity_confidence + 0.1)
        
        return similarity_confidence
    
    def _get_boundary_reason(
        self,
        similarity: float,
        has_marker: bool,
        page_break: bool,
        has_image: bool,
    ) -> str:
        """
        Get human-readable reason for boundary detection.
        
        Args:
            similarity: Semantic similarity
            has_marker: Has structural marker
            page_break: Has page break
            has_image: Has image
            
        Returns:
            Reason string
        """
        reasons = []
        
        if similarity < self.SIMILARITY_THRESHOLD:
            reasons.append(f"low similarity ({similarity:.2f})")
        
        if has_marker:
            reasons.append("structural marker")
        
        if page_break:
            reasons.append("page break")
        
        if has_image:
            reasons.append("has image")
        
        return ", ".join(reasons) if reasons else "unknown"
    
    async def group_chunks_by_product(
        self,
        chunks: List[Dict[str, Any]],
        boundaries: List[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """
        Group chunks into products based on detected boundaries.
        
        Args:
            chunks: List of document chunks
            boundaries: List of boundary detections
            
        Returns:
            List of product groups (each group is a list of chunks)
        """
        if not boundaries:
            # No boundaries, treat all chunks as one product
            return [chunks]
        
        products = []
        current_product = []
        boundary_indices = {b["index"] for b in boundaries}
        
        for i, chunk in enumerate(chunks):
            if i in boundary_indices and current_product:
                # Start new product
                products.append(current_product)
                current_product = [chunk]
            else:
                current_product.append(chunk)
        
        # Add last product
        if current_product:
            products.append(current_product)
        
        logger.info(f"✅ Grouped {len(chunks)} chunks into {len(products)} products")
        
        return products


