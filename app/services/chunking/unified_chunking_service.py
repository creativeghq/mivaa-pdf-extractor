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


class UnifiedChunkingService:
    """
    Unified chunking service that consolidates all chunking strategies.
    
    This service provides:
    - Semantic chunking based on content meaning
    - Fixed-size chunking based on character count
    - Hybrid chunking combining both approaches
    - Layout-aware chunking respecting document structure
    - Consistent chunk metadata and quality scoring
    """
    
    # Sentence endings pattern
    SENTENCE_ENDINGS = r'[.!?]+\s+'
    
    # Paragraph breaks pattern
    PARAGRAPH_BREAKS = r'\n\s*\n'
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize unified chunking service."""
        self.config = config or ChunkingConfig()
        self.logger = logger
    
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
            List of chunks
        """
        try:
            if not text or len(text.strip()) == 0:
                self.logger.warning(f"Empty text provided for document {document_id}")
                return []

            self.logger.info(f"🔄 Starting chunking for document {document_id} ({len(text)} chars) using {self.config.strategy} strategy")

            # Select chunking strategy
            if self.config.strategy == ChunkingStrategy.SEMANTIC:
                self.logger.info(f"   Using SEMANTIC chunking...")
                chunks = self._chunk_semantic(text, document_id, metadata)
            elif self.config.strategy == ChunkingStrategy.FIXED_SIZE:
                self.logger.info(f"   Using FIXED_SIZE chunking...")
                chunks = self._chunk_fixed_size(text, document_id, metadata)
            elif self.config.strategy == ChunkingStrategy.HYBRID:
                self.logger.info(f"   Using HYBRID chunking...")
                chunks = self._chunk_hybrid(text, document_id, metadata)
            elif self.config.strategy == ChunkingStrategy.LAYOUT_AWARE:
                self.logger.info(f"   Using LAYOUT_AWARE chunking...")
                chunks = self._chunk_layout_aware(text, document_id, metadata)
            else:
                raise ValueError(f"Unknown chunking strategy: {self.config.strategy}")

            self.logger.info(f"   Calculating quality scores for {len(chunks)} chunks...")
            # Calculate quality scores for all chunks
            for chunk in chunks:
                chunk.quality_score = self._calculate_chunk_quality(chunk)

            self.logger.info(f"✅ Created {len(chunks)} chunks for document {document_id} using {self.config.strategy} strategy")
            return chunks

        except Exception as e:
            self.logger.error(f"Error chunking text: {e}")
            raise

    async def chunk_pages(
        self,
        pages: List[Dict[str, Any]],
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None
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

        Returns:
            List of chunks with page_number in metadata
        """
        try:
            if not pages or len(pages) == 0:
                self.logger.warning(f"Empty pages list provided for document {document_id}")
                return []

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

                # Chunk this page's text
                page_chunks = self._chunk_page_text(
                    text=page_text,
                    document_id=document_id,
                    page_number=page_number + 1,  # Convert to 1-based for storage
                    start_chunk_index=global_chunk_index,
                    metadata=metadata
                )

                all_chunks.extend(page_chunks)
                global_chunk_index += len(page_chunks)

                if (page_number + 1) % 10 == 0:
                    self.logger.info(f"   Processed {page_number + 1} pages, {len(all_chunks)} chunks so far")

            # Calculate quality scores for all chunks
            self.logger.info(f"   Calculating quality scores for {len(all_chunks)} chunks...")
            for chunk in all_chunks:
                chunk.quality_score = self._calculate_chunk_quality(chunk)

            # Update total_chunks for all chunks
            for chunk in all_chunks:
                chunk.total_chunks = len(all_chunks)

            self.logger.info(f"✅ Page-aware chunking complete: {len(all_chunks)} chunks from {len(pages)} pages")
            return all_chunks

        except Exception as e:
            self.logger.error(f"❌ Page-aware chunking failed: {e}", exc_info=True)
            raise

    def _chunk_page_text(
        self,
        text: str,
        document_id: str,
        page_number: int,
        start_chunk_index: int,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Chunk text from a single page using the configured strategy.

        Args:
            text: Text from the page
            document_id: Document ID
            page_number: Page number (1-based)
            start_chunk_index: Starting index for chunks
            metadata: Additional metadata

        Returns:
            List of chunks with page_number in metadata
        """
        # Select chunking strategy
        if self.config.strategy == ChunkingStrategy.SEMANTIC:
            chunks = self._chunk_semantic(text, document_id, metadata, page_number)
        elif self.config.strategy == ChunkingStrategy.FIXED_SIZE:
            chunks = self._chunk_fixed_size(text, document_id, metadata, page_number)
        elif self.config.strategy == ChunkingStrategy.HYBRID:
            chunks = self._chunk_hybrid(text, document_id, metadata, page_number)
        elif self.config.strategy == ChunkingStrategy.LAYOUT_AWARE:
            chunks = self._chunk_layout_aware(text, document_id, metadata, page_number)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.config.strategy}")

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

        current_chunk = ""

        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx % 50 == 0 and para_idx > 0:
                self.logger.info(f"         SEMANTIC: Processed {para_idx}/{len(paragraphs)} paragraphs, {len(chunks)} chunks so far")

            if not paragraph.strip():
                continue

            paragraph_with_break = paragraph + "\n\n"

            # If adding this paragraph would exceed max size, finalize current chunk
            if len(current_chunk) + len(paragraph_with_break) > self.config.max_chunk_size and current_chunk:
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
                current_chunk = overlap_content + paragraph_with_break
                chunk_index += 1
            else:
                current_chunk += paragraph_with_break

            current_position += len(paragraph_with_break)

        # Add final chunk
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
        Layout-aware chunking that respects document structure.

        Respects hierarchy and structure while maintaining semantic boundaries.

        Args:
            text: Text to chunk
            document_id: Document ID
            metadata: Additional metadata
            page_number: Optional page number (1-based) to store in chunk metadata
        """
        # For now, use semantic chunking as base
        # In future, this can be enhanced with layout analysis
        return self._chunk_semantic(text, document_id, metadata, page_number)
    
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


