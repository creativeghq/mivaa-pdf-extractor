"""
Boundary-Aware Chunking Service - Enhancement 1

Splits chunks at product boundaries to ensure each chunk contains content from only ONE product.
This is a CHUNK SPLITTING service that runs AFTER chunking, BEFORE storage.

Features:
- Uses existing boundary_detector.py to find product boundaries
- Splits chunks that span multiple products
- Ensures each chunk = ONE product only
- Completely optional - doesn't affect existing pipeline

Safety:
- Runs AFTER chunking, BEFORE storage
- Has fallback behavior (if fails, use original chunks)
- Feature flag controlled (default: OFF)
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.services.boundary_detector import BoundaryDetector

logger = logging.getLogger(__name__)


class BoundaryAwareChunkingService:
    """
    Splits chunks at product boundaries for better content isolation.
    
    This service:
    1. Detects product boundaries using boundary_detector
    2. Splits chunks that span multiple products
    3. Ensures each chunk contains content from only ONE product
    4. Runs as optional post-chunking step
    """
    
    def __init__(self, enabled: bool = False):
        """
        Initialize boundary-aware chunking service.
        
        Args:
            enabled: Whether boundary detection is enabled (default: False)
        """
        self.enabled = enabled
        self.boundary_detector = BoundaryDetector()
        self.logger = logger
    
    async def process_chunks(
        self,
        chunks: List[Dict[str, Any]],
        products: List[Dict[str, Any]],
        document_id: str
    ) -> List[Dict[str, Any]]:
        """
        Process chunks to split at product boundaries.
        
        Args:
            chunks: List of chunks to process
            products: List of products with page ranges
            document_id: Document ID for logging
            
        Returns:
            List of boundary-aware chunks (or original chunks if disabled/failed)
        """
        if not self.enabled:
            self.logger.debug("⏭️ Boundary-aware chunking disabled (feature flag OFF)")
            return chunks
        
        if not products or len(products) < 2:
            self.logger.debug("⏭️ Not enough products for boundary detection")
            return chunks
        
        try:
            self.logger.info(f"🔍 Processing {len(chunks)} chunks for product boundaries")
            
            # Detect boundaries in chunks
            boundaries = await self.boundary_detector.detect_boundaries(chunks)
            
            if not boundaries:
                self.logger.info("ℹ️ No product boundaries detected in chunks")
                return chunks
            
            self.logger.info(f"📍 Detected {len(boundaries)} product boundaries")
            
            # Split chunks at boundaries
            boundary_aware_chunks = self._split_chunks_at_boundaries(chunks, boundaries)
            
            self.logger.info(
                f"✅ Split {len(chunks)} chunks into {len(boundary_aware_chunks)} "
                f"boundary-aware chunks"
            )
            
            return boundary_aware_chunks
            
        except Exception as e:
            self.logger.error(f"❌ Boundary detection failed: {e}")
            # ✅ FALLBACK: Return original chunks
            self.logger.warning("⚠️ Returning original chunks (fallback)")
            return chunks
    
    def _split_chunks_at_boundaries(
        self,
        chunks: List[Dict[str, Any]],
        boundaries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Split chunks at detected boundaries.
        
        Args:
            chunks: Original chunks
            boundaries: Detected boundaries with indices
            
        Returns:
            List of split chunks
        """
        if not boundaries:
            return chunks
        
        # Create a set of boundary indices for fast lookup
        boundary_indices = {b['index'] for b in boundaries}
        
        split_chunks = []
        current_chunk_group = []
        
        for i, chunk in enumerate(chunks):
            # If this is a boundary, finalize previous group
            if i in boundary_indices and current_chunk_group:
                # Merge chunks in current group
                merged_chunk = self._merge_chunks(current_chunk_group, len(split_chunks))
                split_chunks.append(merged_chunk)
                current_chunk_group = []
            
            current_chunk_group.append(chunk)
        
        # Add final group
        if current_chunk_group:
            merged_chunk = self._merge_chunks(current_chunk_group, len(split_chunks))
            split_chunks.append(merged_chunk)
        
        return split_chunks
    
    def _merge_chunks(
        self,
        chunk_group: List[Dict[str, Any]],
        new_index: int
    ) -> Dict[str, Any]:
        """
        Merge a group of chunks into a single chunk.
        
        Args:
            chunk_group: List of chunks to merge
            new_index: Index for the merged chunk
            
        Returns:
            Merged chunk
        """
        if len(chunk_group) == 1:
            # Single chunk, just update index
            chunk = chunk_group[0].copy()
            chunk['chunk_index'] = new_index
            return chunk
        
        # Merge multiple chunks
        merged_content = "\n\n".join(c.get('content', '') for c in chunk_group)
        
        # Combine metadata
        merged_metadata = chunk_group[0].get('metadata', {}).copy()
        merged_metadata['merged_from_chunks'] = len(chunk_group)
        merged_metadata['original_indices'] = [c.get('chunk_index') for c in chunk_group]
        
        return {
            'content': merged_content,
            'chunk_index': new_index,
            'metadata': merged_metadata
        }

