"""
Unified Chunking Service.

Consolidates chunking strategies into a single service:
1. Semantic - paragraph/sentence boundaries
2. Fixed-size - character count with word-boundary fallback
3. Hybrid - semantic first, fixed-size for oversized chunks

Layout-aware chunking is NOT a strategy; it's activated by passing
`layout_regions_by_page` into `chunk_pages()`, which routes through
`_chunk_with_layout_regions` to respect YOLO region boundaries.
"""

import logging
import hashlib
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from app.config.confidence_thresholds import ConfidenceThresholds

logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """Supported chunking strategies.

    Layout-aware chunking is activated not by picking a strategy, but by passing
    `layout_regions_by_page` into `chunk_pages()`. See `_chunk_with_layout_regions`.
    """
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"
    HYBRID = "hybrid"


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
    low_quality_rejected: int = 0
    final_chunks: int = 0


class UnifiedChunkingService:
    """
    Unified chunking service.

    Strategies:
    - Semantic: content-meaning boundaries
    - Fixed-size: character count with word-boundary fallback
    - Hybrid: semantic + fixed-size for oversized segments

    Layout-aware chunking is activated per-call by passing layout_regions_by_page
    to chunk_pages(); it routes through `_chunk_with_layout_regions` and respects
    YOLO region boundaries, keeps TABLE regions atomic, and combines TITLE+TEXT.
    """

    SENTENCE_ENDINGS = r'[.!?]+\s+'
    PARAGRAPH_BREAKS = r'\n\s*\n'

    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize unified chunking service."""
        self.config = config or ChunkingConfig()
        self.logger = logger
        self.quality_metrics = ChunkQualityMetrics()
        self._content_hashes: set = set()  # Track exact duplicates

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

            # Detect and annotate cross-references between chunks
            filtered_chunks = self._detect_cross_references(filtered_chunks)

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
        # Use layout-aware chunking if regions are provided AND any of them
        # actually carry `text_content`. YOLO regions ship with bounding
        # boxes but not text — `text_content` only gets populated when an
        # upstream OCR / PDF-text-by-bbox stage runs first. If we hand the
        # layout-aware path regions with all-empty text_content, every
        # region gets skipped (line ~819) and the page produces zero
        # chunks despite having valid text. Bug-D recurrence on every
        # fresh job — see VALENOVA / job b7d70de1.
        regions_have_text = bool(layout_regions) and any(
            (r.get('text_content') or '').strip()
            for r in layout_regions
        )

        if regions_have_text:
            chunks = self._chunk_with_layout_regions(
                text=text,
                document_id=document_id,
                page_number=page_number,
                layout_regions=layout_regions,
                metadata=metadata
            )
        else:
            if layout_regions:
                self.logger.debug(
                    f"   Page {page_number}: {len(layout_regions)} layout "
                    f"region(s) provided but none carry text_content — "
                    f"falling back to text-based chunking"
                )
            # Select and execute chunking strategy on the full page text
            chunks = self._select_chunking_strategy(text, document_id, metadata, page_number)

        # Defensive fallback: if layout-aware path produced 0 chunks but
        # we have a valid page-text payload, retry with text-based
        # chunking. Layout regions can't reliably outproduce a known-good
        # text input.
        if not chunks and text and text.strip():
            self.logger.warning(
                f"   Page {page_number}: layout-aware chunking returned 0 "
                f"chunks despite {len(text)} chars of input text — falling "
                f"back to text-based chunking"
            )
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
        """Find the nearest sentence boundary in text.

        Priority:
          1. Last sentence-ending punctuation (.!?) followed by whitespace.
          2. Last whitespace character — avoids mid-word splits when no
             sentence ending exists in the window.
          3. Full length — only as a last resort for text with no whitespace.
        """
        matches = list(re.finditer(self.SENTENCE_ENDINGS, text))
        if matches:
            return matches[-1].end()
        # Fall back to the last whitespace so we at least split on a word boundary.
        last_ws = max(text.rfind(' '), text.rfind('\n'), text.rfind('\t'))
        if last_ws > 0:
            return last_ws + 1
        return len(text)
    
    def _get_overlap_content(self, text: str, overlap_size: int) -> str:
        """Get overlap content from the end of text."""
        if len(text) <= overlap_size:
            return text
        return text[-overlap_size:]
    
    def _calculate_chunk_quality(self, chunk: Chunk) -> float:
        """Calculate quality score for a chunk.

        Scores four signals:
          - length: fraction of max_chunk_size used (20%)
          - end boundary: ends with sentence punctuation (30%)
          - start boundary: starts cleanly (capital / digit / bullet), not
            mid-word lowercase (20%) — catches splits like "acy Garcia..."
          - semantic completeness: sentence count proxy (30%)
        """
        try:
            content = chunk.content.strip()
            if not content:
                return 0.0

            length_score = min(1.0, len(content) / self.config.max_chunk_size)

            ends_with_punctuation = content.endswith(('.', '!', '?', ':', '"', ')', ']'))
            end_score = 1.0 if ends_with_punctuation else 0.7

            # Start boundary: good if first char is uppercase, digit, or a
            # list/heading marker. Bad if it's a lowercase letter — a strong
            # signal of a mid-word or mid-sentence split.
            first_char = content[0]
            if first_char.isupper() or first_char.isdigit() or first_char in '#•-*"([{':
                start_score = 1.0
            elif first_char.islower():
                start_score = 0.4
            else:
                start_score = 0.8

            sentences = content.count('.') + content.count('!') + content.count('?')
            semantic_score = min(1.0, sentences / 3)

            quality_score = (
                length_score * 0.20 +
                end_score * 0.30 +
                start_score * 0.20 +
                semantic_score * 0.30
            )

            return round(quality_score, 3)
        except Exception:
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

    # Cross-reference patterns: "see page 12", "refer to table 3", "figure 2-1", etc.
    _XREF_PATTERNS: List[Tuple[str, str]] = [
        (r'(?:see|refer(?:ence)?|refer to|shown in|described in|detailed in|as in)\s+page[s]?\s+(\d+)', 'page'),
        (r'(?:see|refer(?:ence)?|refer to|shown in|see also)\s+(?:the\s+)?(?:table|tbl\.?)\s+([\d\.\-]+)', 'table'),
        (r'(?:see|refer(?:ence)?|shown in|as in|refer to)\s+(?:the\s+)?(?:figure|fig\.?)\s+([\d\.\-]+)', 'figure'),
        (r'(?:see|refer to|described in|detailed in)\s+(?:the\s+)?(?:section|sec\.?)\s+([\d\.\-]+)', 'section'),
        (r'(?:see|refer to)\s+(?:the\s+)?(?:appendix|app\.?)\s+([A-Z\d]+)', 'appendix'),
        (r'\((?:see|cf\.?)\s+(?:page[s]?\s+)?(\d+)\)', 'page'),
    ]

    def _detect_cross_references(self, chunks: List['Chunk']) -> List['Chunk']:
        """
        Scan chunks for cross-references (e.g. "See page 12", "Refer to Table 3")
        and annotate each chunk's metadata with resolved target chunk IDs.

        Runs in O(n) over chunks — one regex pass per chunk, one lookup pass to resolve.
        """
        # Build page → chunk_ids index for fast resolution
        page_to_chunk_ids: Dict[int, List[str]] = {}
        for chunk in chunks:
            page_num = chunk.metadata.get('page_number')
            if page_num is not None:
                page_to_chunk_ids.setdefault(int(page_num), []).append(chunk.id)

        compiled = [(re.compile(pat, re.IGNORECASE), ref_type) for pat, ref_type in self._XREF_PATTERNS]

        for chunk in chunks:
            found_refs: List[Dict[str, Any]] = []
            for pattern, ref_type in compiled:
                for match in pattern.finditer(chunk.content):
                    raw_ref = match.group(0)
                    ref_value = match.group(1)

                    target_chunk_ids: List[str] = []
                    if ref_type == 'page':
                        try:
                            target_page = int(ref_value)
                            target_chunk_ids = page_to_chunk_ids.get(target_page, [])
                        except ValueError:
                            pass

                    found_refs.append({
                        'type': ref_type,
                        'raw_text': raw_ref,
                        'ref_value': ref_value,
                        'target_chunk_ids': target_chunk_ids,
                        'resolved': len(target_chunk_ids) > 0,
                    })

            if found_refs:
                chunk.metadata['cross_references'] = found_refs
                self.logger.debug(
                    f"   Chunk {chunk.chunk_index}: found {len(found_refs)} cross-reference(s)"
                )

        total_with_refs = sum(1 for c in chunks if c.metadata.get('cross_references'))
        if total_with_refs:
            self.logger.info(f"   ✅ Cross-references detected: {total_with_refs} chunks have references")
        return chunks

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
                chunk_len = len(current_chunk_text)
                chunk = self._create_chunk(
                    content=current_chunk_text.strip(),
                    chunk_index=chunk_index,
                    document_id=document_id,
                    start_position=current_position,
                    page_number=page_number,
                    metadata={
                        **(metadata or {}),
                        'layout_aware': True,
                        'region_types': list({r.get('region_type') for r in current_chunk_regions}),
                    },
                )
                chunks.append(chunk)
                chunk_index += 1
                current_position += chunk_len
                current_chunk_text = ""
                current_chunk_regions = []

            # Add region to current chunk
            if current_chunk_text:
                current_chunk_text += "\n\n"  # Separate regions with double newline
            current_chunk_text += text_content
            current_chunk_regions.append(region)

            # For TABLE regions, immediately save as complete chunk
            if region_type == 'TABLE':
                chunk_len = len(current_chunk_text)
                chunk = self._create_chunk(
                    content=current_chunk_text.strip(),
                    chunk_index=chunk_index,
                    document_id=document_id,
                    start_position=current_position,
                    page_number=page_number,
                    metadata={
                        **(metadata or {}),
                        'layout_aware': True,
                        'region_types': ['TABLE'],
                        'is_table': True,
                    },
                )
                chunks.append(chunk)
                chunk_index += 1
                current_position += chunk_len
                current_chunk_text = ""
                current_chunk_regions = []

        # Save final chunk if any text remains
        if current_chunk_text:
            chunk = self._create_chunk(
                content=current_chunk_text.strip(),
                chunk_index=chunk_index,
                document_id=document_id,
                start_position=current_position,
                page_number=page_number,
                metadata={
                    **(metadata or {}),
                    'layout_aware': True,
                    'region_types': list({r.get('region_type') for r in current_chunk_regions}),
                },
            )
            chunks.append(chunk)

        self.logger.debug(f"   Created {len(chunks)} layout-aware chunks from {len(sorted_regions)} regions")
        return chunks


