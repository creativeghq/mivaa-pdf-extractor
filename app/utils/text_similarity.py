"""
Text Similarity Utilities

Shared utilities for calculating text similarity across different services.
Consolidates duplicate similarity calculation logic.
"""

import logging
from typing import Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


def calculate_string_similarity(str1: str, str2: str, case_sensitive: bool = False) -> float:
    """
    Calculate similarity between two strings using sequence matching.
    
    Args:
        str1: First string
        str2: Second string
        case_sensitive: Whether to perform case-sensitive comparison
        
    Returns:
        Similarity score between 0.0 and 1.0
        
    Examples:
        >>> calculate_string_similarity("hello world", "hello world")
        1.0
        >>> calculate_string_similarity("hello", "hallo")
        0.8
        >>> calculate_string_similarity("abc", "xyz")
        0.0
    """
    if not str1 or not str2:
        return 0.0
    
    # Normalize strings
    s1 = str1 if case_sensitive else str1.lower().strip()
    s2 = str2 if case_sensitive else str2.lower().strip()
    
    # Use SequenceMatcher for fuzzy string matching
    return SequenceMatcher(None, s1, s2).ratio()


def calculate_text_similarity(
    text1: str,
    text2: str,
    case_sensitive: bool = False,
    method: str = "sequence"
) -> float:
    """
    Calculate semantic similarity between two text blocks.
    
    Args:
        text1: First text block
        text2: Second text block
        case_sensitive: Whether to perform case-sensitive comparison
        method: Similarity method ('sequence', 'word_overlap')
        
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if not text1 or not text2:
        return 0.0
    
    if method == "sequence":
        # Use sequence matching (good for similar text)
        return calculate_string_similarity(text1, text2, case_sensitive)
    
    elif method == "word_overlap":
        # Use word overlap (good for semantic similarity)
        t1 = text1 if case_sensitive else text1.lower()
        t2 = text2 if case_sensitive else text2.lower()
        
        words1 = set(t1.split())
        words2 = set(t2.split())
        
        if not words1 or not words2:
            return 0.0
        
        common_words = words1 & words2
        total_words = max(len(words1), len(words2))
        
        return len(common_words) / total_words if total_words > 0 else 0.0
    
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def is_fuzzy_match(
    str1: str,
    str2: str,
    threshold: float = 0.8,
    case_sensitive: bool = False
) -> bool:
    """
    Check if two strings are fuzzy matches above a threshold.
    
    Args:
        str1: First string
        str2: Second string
        threshold: Minimum similarity score (0.0 to 1.0)
        case_sensitive: Whether to perform case-sensitive comparison
        
    Returns:
        True if similarity >= threshold, False otherwise
        
    Examples:
        >>> is_fuzzy_match("hello world", "hello world", 0.9)
        True
        >>> is_fuzzy_match("hello", "hallo", 0.9)
        False
        >>> is_fuzzy_match("hello", "hallo", 0.7)
        True
    """
    similarity = calculate_string_similarity(str1, str2, case_sensitive)
    return similarity >= threshold


def normalize_text(text: str, lowercase: bool = True, strip: bool = True) -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Text to normalize
        lowercase: Convert to lowercase
        strip: Strip whitespace
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    result = text
    if strip:
        result = result.strip()
    if lowercase:
        result = result.lower()
    
    return result


