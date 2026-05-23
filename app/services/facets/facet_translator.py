"""
Pretranslate layer for the facet canonicalization pipeline.

Sits between L1 (string normalize) and L2 (Voyage cosine cluster). When the
upstream extractor produces a non-English value despite the L0 prompt rule, or
when the ingest path has no LLM upstream (XML supplier feeds), this layer
translates the value to lowercase English BEFORE it hits the canonicalizer.

Why this matters: the resolve_facet_value RPC's defense-in-depth guard rejects
any non-ASCII normalized value as canonical. Without pretranslate, those rejects
would mean lost data. With pretranslate, the canonical row is always English
even when the raw was Greek/Italian/German/etc.

Cost discipline:
  - ASCII-only values bypass Haiku entirely (no API call).
  - Non-ASCII values are batched in ONE Haiku call per canonicalize_product
    invocation regardless of count.
  - Empty batch = no call at all.

The translation is best-effort: if Haiku fails or returns junk, we fall back
to the raw normalized value and let the RPC's non-ASCII guard reject it as
'rejected_non_english'. That gets surfaced in facet_merge_log for follow-up.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"
_MODEL = "claude-haiku-4-5-20251001"

# Strings purely in basic ASCII printable range are presumed English already.
# Falls through pretranslate (no Haiku cost).
_ASCII_ONLY_RE = re.compile(r'^[\x00-\x7f]+$')


def is_ascii_english(value: str) -> bool:
    return bool(_ASCII_ONLY_RE.match(value or ""))


_TRANSLATE_TOOL = {
    "name": "submit_translations",
    "description": (
        "Translate non-English product attribute values to canonical lowercase "
        "English. Each input is a (facet_key, raw_value) pair from a product "
        "ingest pipeline (lighting / tiles / furniture / bathroom / hardware "
        "catalogs). Preserve identifiers, codes, numerics, brand names verbatim."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "description": "One entry per input item, in the same order as supplied.",
                "items": {
                    "type": "object",
                    "properties": {
                        "facet_key": {"type": "string"},
                        "raw": {"type": "string"},
                        "english": {
                            "type": "string",
                            "description": (
                                "Canonical lowercase English translation. Use established "
                                "vocabulary (e.g. 'white', 'warm white', 'matte', 'metal', "
                                "'oak', 'porcelain', 'wall hung'). If the input is already "
                                "English, return the lowercase form. If it's an identifier, "
                                "code, brand, model number, IP rating, socket code, or "
                                "numeric specification, return it verbatim."
                            ),
                        },
                    },
                    "required": ["facet_key", "raw", "english"],
                },
            }
        },
        "required": ["translations"],
    },
}


def _build_user_prompt(items: List[Tuple[str, str]]) -> str:
    rendered = "\n".join(f"  - facet_key={fk!r} raw={rv!r}" for fk, rv in items)
    return (
        "Translate the following product attribute values to canonical lowercase English.\n"
        "Each item is a (facet_key, raw_value) pair from a product catalog ingest.\n\n"
        "Rules:\n"
        "  1. Return one entry per input, in the same order.\n"
        "  2. Use established English catalog vocabulary (e.g. 'white', 'warm white',\n"
        "     'matte', 'metal', 'oak', 'porcelain', 'walnut', 'wall hung', 'basin faucet').\n"
        "  3. If the raw value is already English, return its lowercase form.\n"
        "  4. PRESERVE VERBATIM (no translation): brand names, model numbers, SKUs,\n"
        "     socket codes (E27, GU10, G9, G13, G5, G4, G53), IP ratings (IP20, IP44,\n"
        "     IP65, IP67), wattages ('60W'), voltages ('230V'), CRI/Ra ratings,\n"
        "     dimensions, color temperatures ('2700K', '4000K'), certifications.\n"
        "  5. For colors, use simple English names ('white', not 'snow white' unless\n"
        "     that's the actual catalog distinction).\n"
        "  6. For materials, use the standard catalog term ('metal' not 'metallic',\n"
        "     'wood' not 'wooden', 'glass' not 'glazed').\n"
        "  7. For finishes, use single-word forms ('matte', 'glossy', 'satin',\n"
        "     'brushed', 'polished').\n"
        "  8. Never invent. If you cannot confidently translate, return the raw\n"
        "     value verbatim and let downstream handle it.\n\n"
        f"Items ({len(items)}):\n{rendered}\n\n"
        "Call submit_translations with the full list."
    )


async def translate_facet_values(
    items: List[Tuple[str, str]],
    *,
    timeout_seconds: float = 30.0,
) -> Dict[Tuple[str, str], str]:
    """
    Batched non-English → English translation.

    Args:
      items: list of (facet_key, raw_value) pairs.

    Returns:
      Dict keyed by the input pair → translated English string. ASCII-only inputs
      get mapped to themselves (lowercased) without an API call. Failures leave
      the input absent from the dict — caller should treat that as "no translation"
      and use the raw value (where the RPC's non-ASCII guard will catch it).
    """
    if not items:
        return {}

    # Partition: ASCII shortcuts vs needs-Haiku
    out: Dict[Tuple[str, str], str] = {}
    needs_translation: List[Tuple[str, str]] = []
    for fk, rv in items:
        if rv is None:
            continue
        if is_ascii_english(rv):
            out[(fk, rv)] = rv.strip().lower()
        else:
            needs_translation.append((fk, rv))

    if not needs_translation:
        return out

    api_key = os.getenv("ANTHROPIC_API_KEY") or ""
    if not api_key:
        logger.info("facet-translator: ANTHROPIC_API_KEY missing — non-English values will be rejected by RPC guard")
        return out

    payload = {
        "model": _MODEL,
        "max_tokens": min(4096, 60 + 40 * len(needs_translation)),
        "tools": [_TRANSLATE_TOOL],
        "tool_choice": {"type": "tool", "name": "submit_translations"},
        "messages": [{"role": "user", "content": _build_user_prompt(needs_translation)}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    started = time.time()
    raw_translations: List[dict] = []
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=10.0)) as client:
            resp = await client.post(_ANTHROPIC_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        for block in (data.get("content") or []):
            if block.get("type") == "tool_use" and block.get("name") == "submit_translations":
                raw_translations = (block.get("input") or {}).get("translations") or []
                break
    except Exception as e:
        logger.warning(
            f"facet-translator Haiku call failed after {int((time.time()-started)*1000)}ms: {e}. "
            f"Non-English values will fall through to RPC guard."
        )
        return out

    # Index by (facet_key, raw) so we tolerate the model returning items out of order
    by_key: Dict[Tuple[str, str], str] = {}
    for entry in raw_translations:
        fk = entry.get("facet_key")
        rv = entry.get("raw")
        en = entry.get("english")
        if isinstance(fk, str) and isinstance(rv, str) and isinstance(en, str) and en.strip():
            translated = en.strip().lower()
            # Sanity: if Haiku returned a non-ASCII translation, drop it (RPC guard would reject anyway).
            if is_ascii_english(translated):
                by_key[(fk, rv)] = translated

    out.update(by_key)
    return out
