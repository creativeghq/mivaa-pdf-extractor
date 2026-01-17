"""
Unified Chunking Service - Step 6 Implementation

Consolidates all chunking strategies into a single, unified service:
1. Semantic chunking - Based on content meaning and boundaries
2. Fixed-size chunking - Based on character/token count
3. Hybrid chunking - Combines semantic and fixed-size
4. Layout-aware chunking - Respects document structure

Replaces multiple chunking implementations with a single unified approach.
"""

import logging
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from app.config.confidence_thresholds import ConfidenceThresholds

logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """Supported chunking strategies."""
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"
    HYBRID = "hybrid"
    LAYOUT_AWARE = "layout_aware"


@dataclass
class ChunkingConfig:
    """Configuration for chunking."""
    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    max_chunk_size: int = 1000  # Characters
    min_chunk_size: int = 100   # Characters
    overlap_size: int = 100     # Characters
    preserve_structure: bool = True
    split_on_sentences: bool = True
    split_on_paragraphs: bool = True
    respect_hierarchy: bool = True


@dataclass
class Chunk:
    """Represents a single chunk of text."""
    id: str
    content: str
    chunk_index: int
    total_chunks: int
    start_position: int
    end_position: int
    metadata: Dict[str, Any]
    quality_score: float = 0.0


@dataclass
class ChunkQualityMetrics:
    """Tracks chunk quality metrics during processing."""
    total_chunks_created: int = 0
    exact_duplicates_prevented: int = 0
    semantic_duplicates_prevented: int = 0
    low_quality_rejected: int = 0
    final_chunks: int = 0


class UnifiedChunkingService:
    """
    Unified chunking service that consolidates all chunking strategies.

    This service provides:
    - Semantic chunking based on content meaning
    - Fixed-size chunking based on character count
    - Hybrid chunking combining both approaches
    - Layout-aware chunking respecting document structure
    - Consistent chunk metadata and quality scoring
    - Duplicate detection and quality filtering
    """

    # Sentence endings pattern
    SENTENCE_ENDINGS = r'[.!?]+\s+'

    # Paragraph breaks pattern
    PARAGRAPH_BREAKS = r'\n\s*\n'

    # Similarity threshold for semantic duplicates (cosine similarity)
    SEMANTIC_DUPLICATE_THRESHOLD = 0.95

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize unified chunking service."""
        self.config = config or ChunkingConfig()
        self.logger = logger
        self.quality_metrics = ChunkQualityMetrics()
        self._content_hashes: set = set()  # Track exact duplicates

        # ⚡ OPTIMIZATION: Cache layout regions per product to avoid repeated DB queries
        # Key: product_id, Value: Dict[page_number, List[regions]] or List[regions] for all pages
        self._layout_regions_cache: Dict[str, Dict[Optional[int], List[Dict[str, Any]]]] = {}

        # Get quality threshold from centralized configuration
        self.min_quality_threshold = ConfidenceThresholds.get_threshold(
            "chunking_quality",
            "minimum_acceptable"
        )
    
    async def chunk_text(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk text using the configured strategy.

        Args:
            text: Text to chunk
            document_id: Document ID for chunk metadata
            metadata: Additional metadata for chunks

        Returns:
            List of chunks (filtered for duplicates and quality)
        """
        try:
            if not text or len(text.strip()) == 0:
                self.logger.warning(f"Empty text provided for document {document_id}")
                return []

            # Reset metrics for new document
            self.reset_quality_metrics()

            self.logger.info(f"🔄 Starting chunking for document {document_id} ({len(text)} chars) using {self.config.strategy} strategy")

            # Select and execute chunking strategy
            self.logger.info(f"   Using {self.config.strategy.value} chunking...")
            chunks = self._select_chunking_strategy(text, document_id, metadata)

            self.logger.info(f"   Calculating quality scores for {len(chunks)} chunks...")
            # Calculate quality scores for all chunks
            for chunk in chunks:
                chunk.quality_score = self._calculate_chunk_quality(chunk)

            # Filter for duplicates and quality
            self.logger.info(f"   Filtering chunks for duplicates and quality...")
            filtered_chunks = self.filter_chunks(chunks)

            self.logger.info(f"✅ Created {len(filtered_chunks)} chunks for document {document_id} using {self.config.strategy} strategy")
            return filtered_chunks

        except Exception as e:
            self.logger.error(f"Error chunking text: {e}")
            raise

    async def chunk_pages(
        self,
        pages: List[Dict[str, Any]],
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        layout_regions_by_page: Optional[Dict[int, List[Dict[str, Any]]]] = None
    ) -> List[Chunk]:
        """
        Chunk text from pages while preserving page information.

        This is the PREFERRED method for PDF chunking as it preserves page metadata
        from PyMuPDF4LLM's page_chunks output.

        Args:
            pages: List of page dicts from PyMuPDF4LLM with structure:
                   [{"metadata": {"page": 0}, "text": "..."}, ...]
            document_id: Document ID for chunk metadata
            metadata: Additional metadata for chunks
            layout_regions_by_page: Optional dict mapping page_number -> list of YOLO layout regions
                                   for layout-aware chunking

        Returns:
            List of chunks with page_number in metadata (filtered for duplicates and quality)
        """
        try:
            if not pages or len(pages) == 0:
                self.logger.warning(f"Empty pages list provided for document {document_id}")
                return []

            # Reset metrics for new document
            self.reset_quality_metrics()

            # Check if layout-aware chunking is enabled
            use_layout_aware = layout_regions_by_page is not None and len(layout_regions_by_page) > 0
            if use_layout_aware:
                self.logger.info(f"🔄 Starting LAYOUT-AWARE chunking for document {document_id} ({len(pages)} pages)")
            else:
                self.logger.info(f"🔄 Starting page-aware chunking for document {document_id} ({len(pages)} pages) using {self.config.strategy} strategy")

            all_chunks = []
            global_chunk_index = 0

            # Process each page
            for page_dict in pages:
                page_metadata = page_dict.get('metadata', {})
                page_number = page_metadata.get('page', 0)  # PyMuPDF4LLM uses 0-based indexing
                page_text = page_dict.get('text', '')

                if not page_text or len(page_text.strip()) == 0:
                    self.logger.debug(f"   Skipping empty page {page_number}")
                    continue

                # Get layout regions for this page (1-based page number)
                page_regions = None
                if use_layout_aware:
                    page_regions = layout_regions_by_page.get(page_number + 1, [])
                    if page_regions:
                        self.logger.debug(f"   Page {page_number + 1}: Using {len(page_regions)} layout regions")

                # Chunk this page's text (with optional layout regions)
                page_chunks = self._chunk_page_text(
                    text=page_text,
                    document_id=document_id,
                    page_number=page_number + 1,  # Convert to 1-based for storage
                    start_chunk_index=global_chunk_index,
                    metadata=metadata,
                    layout_regions=page_regions  # Pass layout regions for this page
                )

                all_chunks.extend(page_chunks)
                global_chunk_index += len(page_chunks)

                if (page_number + 1) % 10 == 0:
                    self.logger.info(f"   Processed {page_number + 1} pages, {len(all_chunks)} chunks so far")

            # Calculate quality scores for all chunks
            self.logger.info(f"   Calculating quality scores for {len(all_chunks)} chunks...")
            for chunk in all_chunks:
                chunk.quality_score = self._calculate_chunk_quality(chunk)

            # Filter for duplicates and quality
            self.logger.info(f"   Filtering chunks for duplicates and quality...")
            filtered_chunks = self.filter_chunks(all_chunks)

            # Update total_chunks for all chunks
            for chunk in filtered_chunks:
                chunk.total_chunks = len(filtered_chunks)

            self.logger.info(f"✅ Page-aware chunking complete: {len(filtered_chunks)} chunks from {len(pages)} pages")
            return filtered_chunks

        except Exception as e:
            self.logger.error(f"❌ Page-aware chunking failed: {e}", exc_info=True)
            raise

    def _select_chunking_strategy(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        page_number: Optional[int] = None
    ) -> List[Chunk]:
        """
        Select and execute chunking strategy based on configuration.

        This is the single source of truth for strategy selection.

        Args:
            text: Text to chunk
            document_id: Document ID for chunk metadata
            metadata: Additional metadata for chunks
            page_number: Optional page number for page-aware chunking

        Returns:
            List of chunks
        """
        if self.config.strategy == ChunkingStrategy.SEMANTIC:
            return self._chunk_semantic(text, document_id, metadata, page_number)
        elif self.config.strategy == ChunkingStrategy.FIXED_SIZE:
            return self._chunk_fixed_size(text, document_id, metadata, page_number)
        elif self.config.strategy == ChunkingStrategy.HYBRID:
            return self._chunk_hybrid(text, document_id, metadata, page_number)
        elif self.config.strategy == ChunkingStrategy.LAYOUT_AWARE:
            return self._chunk_layout_aware(text, document_id, metadata, page_number)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.config.strategy}")

    def _chunk_page_text(
        self,
        text: str,
        document_id: str,
        page_number: int,
        start_chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None,
        layout_regions: Optional[List[Dict[str, Any]]] = None
    ) -> List[Chunk]:
        """
        Chunk text from a single page using the configured strategy.

        Args:
            text: Text from the page
            document_id: Document ID
            page_number: Page number (1-based)
            start_chunk_index: Starting index for chunks
            metadata: Additional metadata
            layout_regions: Optional YOLO layout regions for this page

        Returns:
            List of chunks with page_number in metadata
        """
        # Use layout-aware chunking if regions are provided
        if layout_regions and len(layout_regions) > 0:
            chunks = self._chunk_with_layout_regions(
                text=text,
                document_id=document_id,
                page_number=page_number,
                layout_regions=layout_regions,
                metadata=metadata
            )
        else:
            # Select and execute chunking strategy
            chunks = self._select_chunking_strategy(text, document_id, metadata, page_number)

        # Update chunk indices to be global
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = start_chunk_index + i

        return chunks

    def _chunk_semantic(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        page_number: Optional[int] = None
    ) -> List[Chunk]:
        """
        Semantic chunking based on content meaning and boundaries.

        Splits on paragraph boundaries first, then respects sentence boundaries.

        Args:
            text: Text to chunk
            document_id: Document ID
            metadata: Additional metadata
            page_number: Optional page number (1-based) to store in chunk metadata
        """
        chunks = []
        chunk_index = 0
        current_position = 0

        # Split by paragraphs first
        import re
        self.logger.info(f"         SEMANTIC: Splitting {len(text)} chars by paragraphs...")
        paragraphs = re.split(self.PARAGRAPH_BREAKS, text)
        self.logger.info(f"         SEMANTIC: Found {len(paragraphs)} paragraphs, processing...")

        # ⚡ OPTIMIZATION: Use list accumulation instead of string concatenation
        # Python strings are immutable, so += creates new string objects each time
        # Using list.append() + join() is O(n) vs O(n²) for repeated concatenation
        current_chunk_parts: List[str] = []
        current_chunk_length = 0

        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx % 50 == 0 and para_idx > 0:
                self.logger.info(f"         SEMANTIC: Processed {para_idx}/{len(paragraphs)} paragraphs, {len(chunks)} chunks so far")

            if not paragraph.strip():
                continue

            para_length = len(paragraph) + 2  # +2 for "\n\n"

            # If adding this paragraph would exceed max size, finalize current chunk
            if current_chunk_length + para_length > self.config.max_chunk_size and current_chunk_parts:
                # Join accumulated parts into chunk content
                current_chunk = "\n\n".join(current_chunk_parts)
                chunk = self._create_chunk(
                    current_chunk.strip(),
                    document_id,
                    chunk_index,
                    current_position,
                    metadata,
                    page_number
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_content = self._get_overlap_content(current_chunk, self.config.overlap_size)
                current_chunk_parts = [overlap_content, paragraph] if overlap_content else [paragraph]
                current_chunk_length = len(overlap_content) + para_length if overlap_content else para_length
                chunk_index += 1
            else:
                current_chunk_parts.append(paragraph)
                current_chunk_length += para_length

            current_position += para_length

        # Add final chunk
        if current_chunk_parts:
            current_chunk = "\n\n".join(current_chunk_parts)
            if current_chunk.strip():
                chunk = self._create_chunk(
                    current_chunk.strip(),
                    document_id,
                    chunk_index,
                    current_position,
                    metadata,
                    page_number
                )
                chunks.append(chunk)

        self.logger.info(f"         SEMANTIC: Updating total chunks count for {len(chunks)} chunks...")
        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        self.logger.info(f"         SEMANTIC: Complete - {len(chunks)} chunks created")
        return chunks
    
    def _chunk_fixed_size(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        page_number: Optional[int] = None
    ) -> List[Chunk]:
        """
        Fixed-size chunking based on character count.

        Respects sentence boundaries if configured.

        Args:
            text: Text to chunk
            document_id: Document ID
            metadata: Additional metadata
            page_number: Optional page number (1-based) to store in chunk metadata
        """
        chunks = []
        chunk_index = 0
        current_position = 0

        while current_position < len(text):
            end_position = min(current_position + self.config.max_chunk_size, len(text))
            chunk_content = text[current_position:end_position]

            # Adjust for sentence boundaries if enabled
            if self.config.split_on_sentences and end_position < len(text):
                import re
                adjusted_end = self._find_sentence_boundary(chunk_content)
                if adjusted_end > self.config.min_chunk_size:
                    chunk_content = chunk_content[:adjusted_end]

            if len(chunk_content.strip()) >= self.config.min_chunk_size:
                chunk = self._create_chunk(
                    chunk_content.strip(),
                    document_id,
                    chunk_index,
                    current_position,
                    metadata,
                    page_number
                )
                chunks.append(chunk)
                chunk_index += 1

            # Move to next chunk with overlap
            # FIX: Ensure we always move forward to prevent infinite loop
            advance = len(chunk_content) - self.config.overlap_size
            if advance <= 0:
                # If overlap is too large, just move forward by at least 1 char
                advance = max(1, len(chunk_content) // 2)
            current_position += advance

        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks
    
    def _chunk_hybrid(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        page_number: Optional[int] = None
    ) -> List[Chunk]:
        """
        Hybrid chunking combining semantic and fixed-size approaches.

        Starts with semantic chunking, then applies size constraints.

        Args:
            text: Text to chunk
            document_id: Document ID
            metadata: Additional metadata
            page_number: Optional page number (1-based) to store in chunk metadata
        """
        # Start with semantic chunks
        self.logger.info(f"      HYBRID: Starting semantic chunking phase...")
        semantic_chunks = self._chunk_semantic(text, document_id, metadata, page_number)
        self.logger.info(f"      HYBRID: Created {len(semantic_chunks)} semantic chunks, refining...")
        refined_chunks = []

        for idx, chunk in enumerate(semantic_chunks):
            if idx % 10 == 0:
                self.logger.info(f"      HYBRID: Processing chunk {idx+1}/{len(semantic_chunks)}")

            if len(chunk.content) <= self.config.max_chunk_size:
                refined_chunks.append(chunk)
            else:
                # Split oversized semantic chunks using fixed-size approach
                # Create temporary config for fixed-size chunking
                temp_config = ChunkingConfig(
                    strategy=ChunkingStrategy.FIXED_SIZE,
                    max_chunk_size=self.config.max_chunk_size,
                    min_chunk_size=self.config.min_chunk_size,
                    overlap_size=self.config.overlap_size
                )

                # Temporarily switch strategy
                original_strategy = self.config.strategy
                self.config.strategy = ChunkingStrategy.FIXED_SIZE

                sub_chunks = self._chunk_fixed_size(chunk.content, document_id, metadata, page_number)

                # Restore original strategy
                self.config.strategy = original_strategy

                # Update chunk indices
                for i, sub_chunk in enumerate(sub_chunks):
                    sub_chunk.id = f"{chunk.id}_{i}"
                    sub_chunk.chunk_index = len(refined_chunks)
                    refined_chunks.append(sub_chunk)

        self.logger.info(f"      HYBRID: Updating total chunks count for {len(refined_chunks)} chunks...")
        # Update total chunks count
        for chunk in refined_chunks:
            chunk.total_chunks = len(refined_chunks)

        self.logger.info(f"      HYBRID: Complete - {len(refined_chunks)} refined chunks")
        return refined_chunks
    
    def _chunk_layout_aware(
        self,
        text: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        page_number: Optional[int] = None
    ) -> List[Chunk]:
        """
        Layout-aware chunking that respects document structure using YOLO layout regions.

        This method:
        1. Reads layout regions from database (if product_id in metadata)
        2. Respects region boundaries (no mid-sentence splits)
        3. Keeps tables together as single chunks
        4. Preserves title-content relationships
        5. Uses reading order for chunk sequence

        Args:
            text: Text to chunk
            document_id: Document ID
            metadata: Additional metadata (should contain product_id for layout-aware chunking)
            page_number: Optional page number (1-based) to store in chunk metadata

        Returns:
            List of layout-aware chunks
        """
        # Check if we have product_id for layout-aware chunking
        product_id = metadata.get('product_id') if metadata else None

        if not product_id:
            # Fallback to semantic chunking if no product_id
            self.logger.info("      LAYOUT-AWARE: No product_id found, falling back to semantic chunking")
            return self._chunk_semantic(text, document_id, metadata, page_number)

        try:
            # Fetch layout regions from database
            layout_regions = self._fetch_layout_regions(product_id, page_number)

            if not layout_regions:
                # Fallback to semantic chunking if no layout regions
                self.logger.info(f"      LAYOUT-AWARE: No layout regions found for product {product_id}, falling back to semantic chunking")
                return self._chunk_semantic(text, document_id, metadata, page_number)

            self.logger.info(f"      LAYOUT-AWARE: Found {len(layout_regions)} layout regions for product {product_id}")

            # Create chunks based on layout regions
            chunks = []
            chunk_index = 0

            # Sort regions by reading order
            sorted_regions = sorted(layout_regions, key=lambda r: r.get('reading_order', 999))

            for region in sorted_regions:
                region_type = region.get('region_type')
                text_content = region.get('text_content')

                # Skip regions without text content
                if not text_content or not text_content.strip():
                    continue

                # Create chunk based on region type
                chunk_metadata = {
                    **(metadata or {}),
                    'region_type': region_type,
                    'reading_order': region.get('reading_order'),
                    'bbox': {
                        'x': region.get('bbox_x'),
                        'y': region.get('bbox_y'),
                        'width': region.get('bbox_width'),
                        'height': region.get('bbox_height')
                    },
                    'confidence': region.get('confidence')
                }

                # For TABLE regions, keep entire table together
                if region_type == 'TABLE':
                    chunk = self._create_chunk(
                        content=text_content,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        start_position=0,
                        metadata=chunk_metadata,
                        page_number=page_number
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # For TITLE regions, combine with next TEXT region if possible
                elif region_type == 'TITLE':
                    # Look for next TEXT region
                    next_region_idx = sorted_regions.index(region) + 1
                    if next_region_idx < len(sorted_regions):
                        next_region = sorted_regions[next_region_idx]
                        if next_region.get('region_type') == 'TEXT' and next_region.get('text_content'):
                            # Combine title with text
                            combined_content = f"{text_content}\n\n{next_region.get('text_content')}"
                            chunk = self._create_chunk(
                                content=combined_content,
                                document_id=document_id,
                                chunk_index=chunk_index,
                                start_position=0,
                                metadata=chunk_metadata,
                                page_number=page_number
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                            # Skip next region since we combined it
                            sorted_regions[next_region_idx]['_skip'] = True
                        else:
                            # Just create chunk for title
                            chunk = self._create_chunk(
                                content=text_content,
                                document_id=document_id,
                                chunk_index=chunk_index,
                                start_position=0,
                                metadata=chunk_metadata,
                                page_number=page_number
                            )
                            chunks.append(chunk)
                            chunk_index += 1
                    else:
                        # Last region, just create chunk
                        chunk = self._create_chunk(
                            content=text_content,
                            document_id=document_id,
                            chunk_index=chunk_index,
                            start_position=0,
                            metadata=chunk_metadata,
                            page_number=page_number
                        )
                        chunks.append(chunk)
                        chunk_index += 1

                # For TEXT regions, check if we should skip (combined with title)
                elif region_type == 'TEXT':
                    if region.get('_skip'):
                        continue

                    # Split large TEXT regions using semantic chunking
                    if len(text_content) > self.config.max_chunk_size:
                        sub_chunks = self._chunk_semantic(text_content, document_id, chunk_metadata, page_number)
                        for sub_chunk in sub_chunks:
                            sub_chunk.chunk_index = chunk_index
                            chunks.append(sub_chunk)
                            chunk_index += 1
                    else:
                        chunk = self._create_chunk(
                            content=text_content,
                            document_id=document_id,
                            chunk_index=chunk_index,
                            start_position=0,
                            metadata=chunk_metadata,
                            page_number=page_number
                        )
                        chunks.append(chunk)
                        chunk_index += 1

                # For other region types (IMAGE, CAPTION), create simple chunks
                else:
                    chunk = self._create_chunk(
                        content=text_content,
                        document_id=document_id,
                        chunk_index=chunk_index,
                        start_position=0,
                        metadata=chunk_metadata,
                        page_number=page_number
                    )
                    chunks.append(chunk)
                    chunk_index += 1

            # Update total chunks count
            for chunk in chunks:
                chunk.total_chunks = len(chunks)

            self.logger.info(f"      LAYOUT-AWARE: Created {len(chunks)} layout-aware chunks")
            return chunks

        except Exception as e:
            self.logger.error(f"      LAYOUT-AWARE: Error during layout-aware chunking: {e}")
            self.logger.info("      LAYOUT-AWARE: Falling back to semantic chunking")
            return self._chunk_semantic(text, document_id, metadata, page_number)

    def _fetch_layout_regions(self, product_id: str, page_number: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        ⚡ OPTIMIZED: Fetch layout regions with caching.

        Fetches all regions for a product ONCE, then filters by page from cache.
        This avoids N database queries for N pages.

        Args:
            product_id: Product UUID
            page_number: Optional page number to filter by

        Returns:
            List of layout region dictionaries
        """
        try:
            # 🔥 FIX: Check if product_id is a valid UUID before querying
            import uuid
            try:
                uuid.UUID(product_id)
            except (ValueError, AttributeError):
                self.logger.debug(f"      LAYOUT-AWARE: product_id '{product_id}' is not a UUID, skipping layout regions fetch")
                return []

            # ⚡ OPTIMIZATION: Check cache first
            if product_id in self._layout_regions_cache:
                all_regions = self._layout_regions_cache[product_id].get(None, [])
                if page_number is not None:
                    # Filter from cached data
                    return [r for r in all_regions if r.get('page_number') == page_number]
                return all_regions

            # Fetch ALL regions for this product ONCE (not per-page)
            from app.services.core.supabase_client import get_supabase_client
            supabase = get_supabase_client()

            # Fetch all regions for product (no page filter) and order by reading_order
            query = supabase.client.table('product_layout_regions').select('*').eq('product_id', product_id)
            query = query.order('reading_order')
            result = query.execute()

            all_regions = result.data if result.data else []

            # Store in cache
            self._layout_regions_cache[product_id] = {None: all_regions}
            self.logger.debug(f"      LAYOUT-AWARE: Cached {len(all_regions)} regions for product {product_id}")

            # Filter by page if requested
            if page_number is not None:
                return [r for r in all_regions if r.get('page_number') == page_number]

            return all_regions

        except Exception as e:
            self.logger.error(f"      LAYOUT-AWARE: Failed to fetch layout regions: {e}")
            return []

    def clear_layout_cache(self, product_id: Optional[str] = None):
        """Clear layout regions cache for a product or all products."""
        if product_id:
            self._layout_regions_cache.pop(product_id, None)
        else:
            self._layout_regions_cache.clear()
    
    def _create_chunk(
        self,
        content: str,
        document_id: str,
        chunk_index: int,
        start_position: int,
        metadata: Optional[Dict[str, Any]] = None,
        page_number: Optional[int] = None
    ) -> Chunk:
        """
        Create a chunk with metadata.

        Args:
            content: Chunk content
            document_id: Document ID
            chunk_index: Chunk index
            start_position: Start position in document
            metadata: Additional metadata
            page_number: Optional page number (1-based) to store in chunk metadata
        """
        chunk_id = f"{document_id}_chunk_{chunk_index}_{int(datetime.utcnow().timestamp() * 1000)}"

        chunk_metadata = {
            **(metadata or {}),
            "chunk_strategy": self.config.strategy.value,
            "chunk_size_actual": len(content),
            "created_at": datetime.utcnow().isoformat()
        }

        # Add page_number if provided
        if page_number is not None:
            chunk_metadata["page_number"] = page_number

        return Chunk(
            id=chunk_id,
            content=content,
            chunk_index=chunk_index,
            total_chunks=0,  # Will be updated later
            start_position=start_position,
            end_position=start_position + len(content),
            metadata=chunk_metadata
        )
    
    def _find_sentence_boundary(self, text: str) -> int:
        """Find the nearest sentence boundary in text."""
        import re
        matches = list(re.finditer(self.SENTENCE_ENDINGS, text))
        if matches:
            return matches[-1].end()
        return len(text)
    
    def _get_overlap_content(self, text: str, overlap_size: int) -> str:
        """Get overlap content from the end of text."""
        if len(text) <= overlap_size:
            return text
        return text[-overlap_size:]
    
    def _calculate_chunk_quality(self, chunk: Chunk) -> float:
        """Calculate quality score for a chunk."""
        try:
            # Check content length
            length_score = min(1.0, len(chunk.content) / self.config.max_chunk_size)

            # Check for proper boundaries
            ends_with_punctuation = chunk.content.strip().endswith(('.', '!', '?'))
            boundary_score = 1.0 if ends_with_punctuation else 0.7

            # Check for semantic completeness
            sentences = chunk.content.count('.')
            semantic_score = min(1.0, sentences / 3)  # 3+ sentences = 1.0

            # Weighted average
            quality_score = (
                length_score * 0.3 +
                boundary_score * 0.4 +
                semantic_score * 0.3
            )

            return round(quality_score, 3)
        except:
            return 0.5

    def _get_content_hash(self, content: str) -> str:
        """Generate hash for exact duplicate detection."""
        # Normalize content: lowercase, strip whitespace, remove extra spaces
        normalized = ' '.join(content.lower().strip().split())
        return hashlib.md5(normalized.encode()).hexdigest()

    def _is_exact_duplicate(self, chunk: Chunk) -> bool:
        """Check if chunk is an exact duplicate."""
        content_hash = self._get_content_hash(chunk.content)
        if content_hash in self._content_hashes:
            self.quality_metrics.exact_duplicates_prevented += 1
            self.logger.debug(f"   ⏭️ Skipping exact duplicate chunk (hash: {content_hash[:8]}...)")
            return True
        self._content_hashes.add(content_hash)
        return False

    def _is_low_quality(self, chunk: Chunk) -> bool:
        """Check if chunk quality is below threshold."""
        if chunk.quality_score < self.min_quality_threshold:
            self.quality_metrics.low_quality_rejected += 1
            self.logger.debug(
                f"   ⏭️ Rejecting low quality chunk "
                f"(score: {chunk.quality_score:.3f} < {self.min_quality_threshold})"
            )
            return True
        return False

    def filter_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Filter chunks to remove duplicates and low quality chunks.

        Args:
            chunks: List of chunks to filter

        Returns:
            Filtered list of chunks
        """
        filtered_chunks = []

        for chunk in chunks:
            self.quality_metrics.total_chunks_created += 1

            # Check for exact duplicates
            if self._is_exact_duplicate(chunk):
                continue

            # Check quality threshold
            if self._is_low_quality(chunk):
                continue

            filtered_chunks.append(chunk)

        self.quality_metrics.final_chunks = len(filtered_chunks)

        if self.quality_metrics.exact_duplicates_prevented > 0 or self.quality_metrics.low_quality_rejected > 0:
            self.logger.info(
                f"   🔍 Filtered chunks: {self.quality_metrics.total_chunks_created} → {self.quality_metrics.final_chunks} "
                f"(duplicates: {self.quality_metrics.exact_duplicates_prevented}, "
                f"low quality: {self.quality_metrics.low_quality_rejected})"
            )

        return filtered_chunks

    def get_quality_metrics(self) -> ChunkQualityMetrics:
        """Get current quality metrics."""
        return self.quality_metrics

    def reset_quality_metrics(self):
        """Reset quality metrics for new document."""
        self.quality_metrics = ChunkQualityMetrics()
        self._content_hashes.clear()

    def _chunk_with_layout_regions(
        self,
        text: str,
        document_id: str,
        page_number: int,
        layout_regions: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Create chunks using YOLO layout regions as boundaries.

        This method implements layout-aware chunking that:
        1. Respects region boundaries (don't split TEXT, TITLE, CAPTION regions)
        2. Keeps TABLE regions together
        3. Preserves TITLE + content relationships
        4. Uses reading_order for correct sequence

        Args:
            text: Full page text
            document_id: Document ID
            page_number: Page number (1-based)
            layout_regions: YOLO layout regions for this page
            metadata: Additional metadata

        Returns:
            List of chunks respecting layout boundaries
        """
        chunks = []
        chunk_index = 0

        # Sort regions by reading_order (already sorted from DB query, but ensure it)
        sorted_regions = sorted(layout_regions, key=lambda r: r.get('reading_order', 999))

        # Group regions into semantic chunks
        current_chunk_text = ""
        current_chunk_regions = []
        current_position = 0

        for region in sorted_regions:
            region_type = region.get('region_type', 'TEXT')
            text_content = region.get('text_content', '')

            # Skip empty regions
            if not text_content or len(text_content.strip()) == 0:
                continue

            # Decide whether to start a new chunk
            should_start_new_chunk = False

            # Rule 1: TABLE regions always start a new chunk and stay together
            if region_type == 'TABLE':
                if current_chunk_text:
                    # Save current chunk before starting table
                    should_start_new_chunk = True

            # Rule 2: TITLE regions start a new chunk (but include following content)
            elif region_type == 'TITLE':
                if current_chunk_text:
                    should_start_new_chunk = True

            # Rule 3: Check chunk size limit
            elif len(current_chunk_text) + len(text_content) > self.config.max_chunk_size:
                should_start_new_chunk = True

            # Save current chunk if needed
            if should_start_new_chunk and current_chunk_text:
                chunk = self._create_chunk(
                    content=current_chunk_text.strip(),
                    chunk_index=chunk_index,
                    document_id=document_id,
                    start_position=current_position,
                    end_position=current_position + len(current_chunk_text),
                    metadata={
                        **(metadata or {}),
                        'page_number': page_number,
                        'layout_aware': True,
                        'region_types': list(set(r.get('region_type') for r in current_chunk_regions))
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
                current_chunk_text = ""
                current_chunk_regions = []
                current_position += len(current_chunk_text)

            # Add region to current chunk
            if current_chunk_text:
                current_chunk_text += "\n\n"  # Separate regions with double newline
            current_chunk_text += text_content
            current_chunk_regions.append(region)

            # For TABLE regions, immediately save as complete chunk
            if region_type == 'TABLE':
                chunk = self._create_chunk(
                    content=current_chunk_text.strip(),
                    chunk_index=chunk_index,
                    document_id=document_id,
                    start_position=current_position,
                    end_position=current_position + len(current_chunk_text),
                    metadata={
                        **(metadata or {}),
                        'page_number': page_number,
                        'layout_aware': True,
                        'region_types': ['TABLE'],
                        'is_table': True
                    }
                )
                chunks.append(chunk)
                chunk_index += 1
                current_chunk_text = ""
                current_chunk_regions = []
                current_position += len(current_chunk_text)

        # Save final chunk if any text remains
        if current_chunk_text:
            chunk = self._create_chunk(
                content=current_chunk_text.strip(),
                chunk_index=chunk_index,
                document_id=document_id,
                start_position=current_position,
                end_position=current_position + len(current_chunk_text),
                metadata={
                    **(metadata or {}),
                    'page_number': page_number,
                    'layout_aware': True,
                    'region_types': list(set(r.get('region_type') for r in current_chunk_regions))
                }
            )
            chunks.append(chunk)

        self.logger.debug(f"   Created {len(chunks)} layout-aware chunks from {len(sorted_regions)} regions")
        return chunks


