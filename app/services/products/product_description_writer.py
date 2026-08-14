"""
Product description writer — Claude Haiku over product chunks.

Why this exists:
  The Stage 0 AI metadata extractor is designed to output structured fields.
  It routinely leaves products.description empty because there's no explicit
  "description" column in tile catalogs — the description is prose scattered
  across multiple chunks, interleaved with page markers, bilingual copy, and
  SKU tables.

  This module takes ALL of a product's chunks, sends them to Claude Haiku with
  a tight prompt asking for a clean 2-4 sentence English description, and
  writes the result to products.description.

Cost: ~$0.0003 per product (small Haiku call). Runs inside Stage 4.7 and
also via the backfill endpoint.
"""

import logging
import os
import re
from typing import Any, List, Optional

from app.services.core.anthropic_error_reporter import report_anthropic_failure
from app.services.utilities.prompt_registry import get_cached

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

DESCRIPTION_MODEL = os.getenv("PRODUCT_DESCRIPTION_MODEL", "claude-haiku-4-5")

MAX_INPUT_CHARS = 6000  # ~1500 tokens — plenty for product context without overspend

# Prompt: generation / product_description (#347 phase 3P).



def _clean_chunk_text(text: str) -> str:
    """Light-touch cleaning — remove PDF artifacts without damaging real content."""
    if not text:
        return ""
    # Strip page separators that appear inline
    text = re.sub(r"---\s*#\s*Page\s*\d+\s*---", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Page\s*\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\d+\s*[―—-]", "", text, flags=re.MULTILINE)
    # Collapse runs of whitespace
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _is_likely_english(text: str) -> bool:
    """Quick heuristic: does >40% of the text's words look English?

    Uses a set of common English function words. Spanish/Italian/French
    catalogs use different articles/prepositions (de, la, un, una, con, que,
    para, etc.) so this is a reliable split for bilingual ceramic catalogs
    where one paragraph is in English and the next is the Spanish translation.
    """
    if not text:
        return False
    english_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'of',
        'to', 'in', 'for', 'with', 'that', 'this', 'from', 'by', 'on', 'it',
        'its', 'has', 'have', 'be', 'not', 'but', 'which', 'each', 'can',
        'new', 'our', 'any', 'all', 'will', 'at', 'as', 'their', 'been',
        'only', 'also', 'into', 'creating', 'collection', 'design', 'tile',
    }
    words = re.findall(r'[a-zA-Z]{2,}', text.lower())
    if not words:
        return False
    english_count = sum(1 for w in words if w in english_words)
    return (english_count / len(words)) > 0.15


def write_product_description_from_chunks(
    product_name: str,
    chunks: List[Any],
    *,
    job_id: Optional[str] = None,
    product_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> Optional[str]:
    """Generate a clean English product description from chunks.

    Args:
        product_name: The product's name (used to anchor the Claude prompt).
        chunks: List of chunk dicts or objects with a `.content` field.

    Returns:
        A clean 2-4 sentence English description string, or None if we couldn't
        produce one (no chunks, empty chunks, API failure, or Claude returned
        empty string).
    """
    if not ANTHROPIC_API_KEY:
        logger.warning("product_description_writer: ANTHROPIC_API_KEY not set")
        return None
    if not chunks:
        return None

    # Assemble the raw chunk text (capped).
    #
    # Bilingual ceramic catalogs (Harmony, Peronda, ...) place the Spanish
    # paragraph before the English one on spreads. If we just iterate chunks
    # in document order, the 6000-char cap can fill up on Spanish text and
    # the English description never makes it to Claude. We sort English-
    # looking chunks first so the model always gets the English narrative
    # within the cap, then any leftover budget goes to the Spanish (which
    # Claude can translate if needed).
    cleaned_chunks: List[tuple] = []  # (cleaned_text, is_english)
    for c in chunks:
        if isinstance(c, dict):
            content = c.get("content") or ""
        else:
            content = getattr(c, "content", "") or ""
        cleaned = _clean_chunk_text(str(content))
        if not cleaned:
            continue
        cleaned_chunks.append((cleaned, _is_likely_english(cleaned)))

    # English first, then non-English
    cleaned_chunks.sort(key=lambda x: (not x[1], len(x[0])))

    parts: List[str] = []
    remaining = MAX_INPUT_CHARS
    for cleaned, _is_eng in cleaned_chunks:
        if len(cleaned) > remaining:
            cleaned = cleaned[:remaining]
        parts.append(cleaned)
        remaining -= len(cleaned)
        if remaining <= 0:
            break

    if not parts:
        return None

    chunks_text = "\n\n".join(parts)

    try:
        from app.services.core.claude_helper import tracked_claude_call
        resp = tracked_claude_call(
            task="product_description_write",
            model=DESCRIPTION_MODEL,
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": get_cached("generation", "product_description").format(
                    product_name=product_name or "(unnamed)",
                    chunks_text=chunks_text,
                ),
            }],
            job_id=job_id,
            product_id=product_id,
            workspace_id=workspace_id,
        )
    except Exception as e:
        report_anthropic_failure(
            e,
            service="product_description_writer",
            context={"product_name": product_name},
        )
        logger.warning(f"product_description_writer: Claude call failed: {e}")
        return None

    text = resp.content[0].text if resp.content else ""
    text = text.strip()

    # Strip leading/trailing quotes or markdown
    text = text.strip('"').strip("'").strip("`").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1].strip()

    # Guard against Claude returning prose like "Here is the description:"
    if text.lower().startswith(("here is", "here's", "description:")):
        text = text.split(":", 1)[-1].strip()

    if not text or len(text) < 20:
        logger.info(f"product_description_writer: Claude returned too-short result ({len(text)} chars) for '{product_name}'")
        return None

    # Cap at reasonable length
    if len(text) > 1500:
        text = text[:1500].rstrip() + "…"

    return text
