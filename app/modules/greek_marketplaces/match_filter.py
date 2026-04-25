"""
Match-quality safeguard for the marketplace adapters.

When a search query has no real matches, scrape targets often fall back
to showing featured / suggested products. The Firecrawl extractor will
happily report those as the "first product on the page" — that's how a
faucet query came back with a Brenthaven notebook lock from bestdeals.gr
in earlier testing.

This module catches that class of false positive WITHOUT a second LLM
call: it tokenizes the query and the candidate URL slug, and rejects
results that share zero meaningful tokens. Cheap, deterministic,
language-agnostic (works for both Greek and Latin script).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional
from urllib.parse import unquote, urlparse

# Tokens shorter than this are too generic (articles, units, single chars).
_MIN_TOKEN_LEN = 3

# Greek + Latin stopwords that frequently leak into product queries.
_STOPWORDS = {
    # Greek
    "και", "για", "της", "του", "των", "στη", "στο", "στις", "στους", "από",
    "ένα", "μία", "ενός", "μιας", "οι", "τα", "τη", "τον", "την",
    # Generic units / descriptors
    "cm", "mm", "inch", "x", "kg", "gr", "ml", "lt", "set", "pcs",
    # Common product nouns that would over-match
    "νέο", "new", "item", "product",
}


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s)
        if unicodedata.category(c) != "Mn"
    )


def _tokenize(text: str) -> set[str]:
    """Lowercase, strip accents, split on non-word, drop stopwords + short tokens."""
    if not text:
        return set()
    normalized = _strip_accents(text.lower())
    raw = re.split(r"[^a-z0-9α-ω]+", normalized, flags=re.IGNORECASE)
    return {
        t for t in raw
        if t and len(t) >= _MIN_TOKEN_LEN and t not in _STOPWORDS
    }


def is_plausible_match(
    query: str,
    candidate_url: str,
    candidate_name: Optional[str] = None,
    *,
    min_overlap_ratio: float = 0.30,
) -> bool:
    """
    Return True if the candidate (URL slug + optional product name) shares
    enough tokens with the query to plausibly be the same product.

    `min_overlap_ratio` is the fraction of query tokens that must appear
    somewhere in the candidate (slug ∪ name). 0.30 means ≥30% — strict
    enough to kill "different product entirely" matches, lenient enough
    to allow word-order shuffles, transliteration, and partial titles.
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        # No meaningful query tokens (e.g. all stopwords) — be permissive.
        return True

    # Pull tokens from the URL path/slug. Decode percent-encoded bytes
    # first so 'πλακάκι' shows up as Greek letters not %CF%80%CE%BB%CE%B1...
    parsed = urlparse(candidate_url or "")
    slug = unquote(parsed.path or "")
    candidate_tokens = _tokenize(slug)
    if candidate_name:
        candidate_tokens |= _tokenize(candidate_name)

    if not candidate_tokens:
        return False

    overlap = query_tokens & candidate_tokens
    ratio = len(overlap) / len(query_tokens)
    return ratio >= min_overlap_ratio
