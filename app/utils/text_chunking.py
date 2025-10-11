"""
Text Chunking Utilities

This module provides functions for splitting text into chunks for processing.
"""

from typing import List


def chunk_text(
    text: str, 
    chunk_size: int = 1000, 
    overlap: int = 200,
    separator: str = "\n\n"
) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        separator: Preferred separator for splitting (default: double newline)
        
    Returns:
        List of text chunks
    """
    if not text or chunk_size <= 0:
        return []
    
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position
        end = start + chunk_size
        
        if end >= len(text):
            # Last chunk
            chunks.append(text[start:])
            break
        
        # Try to find a good breaking point
        chunk_end = end
        
        # Look for separator within the chunk
        separator_pos = text.rfind(separator, start, end)
        if separator_pos > start:
            chunk_end = separator_pos + len(separator)
        else:
            # Look for sentence endings
            sentence_end = text.rfind('.', start, end)
            if sentence_end > start:
                chunk_end = sentence_end + 1
            else:
                # Look for any whitespace
                space_pos = text.rfind(' ', start, end)
                if space_pos > start:
                    chunk_end = space_pos
        
        # Extract chunk
        chunk = text[start:chunk_end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Calculate next start position with overlap
        start = max(start + 1, chunk_end - overlap)
    
    return chunks


def smart_chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    min_chunk_size: int = 100
) -> List[str]:
    """
    Smart text chunking that tries to preserve semantic boundaries.
    
    Args:
        text: The text to chunk
        chunk_size: Target size of each chunk in characters
        overlap: Number of characters to overlap between chunks
        min_chunk_size: Minimum size for a chunk to be valid
        
    Returns:
        List of text chunks
    """
    if not text or chunk_size <= 0:
        return []
    
    # First, try to split by paragraphs
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # If adding this paragraph would exceed chunk size
        if len(current_chunk) + len(paragraph) + 2 > chunk_size:
            # Save current chunk if it's not empty
            if current_chunk and len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
            
            # If paragraph itself is too large, split it
            if len(paragraph) > chunk_size:
                para_chunks = chunk_text(paragraph, chunk_size, overlap)
                chunks.extend(para_chunks)
                current_chunk = ""
            else:
                current_chunk = paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add final chunk
    if current_chunk and len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk.strip())
    
    return chunks
