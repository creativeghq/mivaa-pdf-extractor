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

import anthropic

from app.services.core.anthropic_error_reporter import report_anthropic_failure

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

DESCRIPTION_MODEL = os.getenv("PRODUCT_DESCRIPTION_MODEL", "claude-haiku-4-5-20251001")

MAX_INPUT_CHARS = 6000  # ~1500 tokens — plenty for product context without overspend

DESCRIPTION_PROMPT = """You are writing a clean 2-4 sentence English product description for a ceramic tile (or similar surface material) product catalog entry.

Input: raw chunks of text extracted from a PDF page, which may contain:
- the real description paragraphs (in English, sometimes also in Spanish/Italian/French)
- PDF page separators like "--- # Page N ---"
- SKU code lines like "39656 VALENOVA WHITE LT/11,8X11,8"
- packing table artifacts ("PACKING", "BOXES PALLET", "UNIT m2", etc.)
- page numbers like "24  —"
- designer biographies that should NOT be included in the product description

Your job:
1. Extract ONLY the English narrative describing THIS product (name, designer, inspiration, aesthetic, format/size, palette, suggested use).
2. Write it as 2-4 flowing sentences. Natural English, no filler.
3. If the text is only Spanish/Italian/French, translate to English.
4. Do NOT include: SKU codes, page numbers, packing data, designer bios that aren't about the product itself.
5. Do NOT hallucinate fields that aren't in the source. If there's no real description, return an empty string.

Return ONLY the description text (or empty string). No prose, no JSON, no quotes, no markdown.

Product name: {product_name}

Raw chunks:
{chunks_text}

Write the description now:"""


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


def write_product_description_from_chunks(
    product_name: str,
    chunks: List[Any],
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

    # Assemble the raw chunk text (capped)
    parts: List[str] = []
    remaining = MAX_INPUT_CHARS
    for c in chunks:
        if isinstance(c, dict):
            content = c.get("content") or ""
        else:
            content = getattr(c, "content", "") or ""
        cleaned = _clean_chunk_text(str(content))
        if not cleaned:
            continue
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
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=DESCRIPTION_MODEL,
            max_tokens=400,
            messages=[{
                "role": "user",
                "content": DESCRIPTION_PROMPT.format(
                    product_name=product_name or "(unnamed)",
                    chunks_text=chunks_text,
                ),
            }],
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
