"""Pretranslate layer for the facet canonicalization pipeline."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Dict, List, Tuple


logger = logging.getLogger(__name__)

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


async def _build_user_prompt(items: List[Tuple[str, str]]) -> str:
    """The DB prompt with the items fenced as catalogue DATA (invariant 9)."""
    from app.services.utilities.prompt_registry import load_prompt, render
    from app.utils.untrusted_content import as_untrusted_data

    template = await load_prompt("extraction", "facet_translation", stage="canonicalization")
    rendered = "\n".join(f"  - facet_key={fk!r} raw={rv!r}" for fk, rv in items)
    return render(template, item_count=len(items),
                  items=as_untrusted_data(rendered, source="catalogue attribute values"))


async def translate_facet_values(
    items: List[Tuple[str, str]],
    *,
    timeout_seconds: float = 30.0,
) -> Dict[Tuple[str, str], str]:
    """Batched non-English -> English translation; see `translate_facet_values_with_status`."""
    translations, _degraded = await translate_facet_values_with_status(
        items, timeout_seconds=timeout_seconds)
    return translations


async def translate_facet_values_with_status(
    items: List[Tuple[str, str]],
    *,
    timeout_seconds: float = 30.0,
) -> Tuple[Dict[Tuple[str, str], str], bool]:
    """Batched non-English -> English translation, and whether it DEGRADED.

    Returns (translations, degraded). ASCII inputs map to themselves (lowercased) with no
    API call. `degraded` is True when a value that needed translating did not get one —
    no key, a failed call, or a missing/unusable entry — because the RPC then rejects it
    and the facet silently disappears from the product.
    """
    if not items:
        return {}, False

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
        return out, False

    from app.services.integrations.platform_secret_resolver import resolve_secret

    api_key = (await asyncio.to_thread(resolve_secret, "ANTHROPIC_API_KEY")).value or ""
    if not api_key:
        logger.warning(
            "facet-translator: ANTHROPIC_API_KEY unresolved — %d non-English value(s) left "
            "untranslated (degraded)", len(needs_translation))
        return out, True

    # Outside the try: PromptNotConfigured / PromptStoreUnavailable must not read as a model failure.
    user_prompt = await _build_user_prompt(needs_translation)
    started = time.time()
    try:
        from app.services.core.claude_tool_call import call_with_tool
        result = await call_with_tool(
            task="facet_translation",
            model=_MODEL,
            max_tokens=min(4096, 60 + 40 * len(needs_translation)),
            messages=[{"role": "user", "content": user_prompt}],
            tool=_TRANSLATE_TOOL,
        )
        raw_translations = result.data.get("translations") or []
    except Exception as e:
        logger.warning(
            f"facet-translator Haiku call failed after {int((time.time()-started)*1000)}ms: {e}. "
            f"{len(needs_translation)} value(s) left untranslated (degraded)."
        )
        return out, True

    by_key: Dict[Tuple[str, str], str] = {}
    for entry in raw_translations:
        fk = entry.get("facet_key")
        rv = entry.get("raw")
        en = entry.get("english")
        if isinstance(fk, str) and isinstance(rv, str) and isinstance(en, str) and en.strip():
            translated = en.strip().lower()
            if is_ascii_english(translated):
                by_key[(fk, rv)] = translated

    out.update(by_key)
    missing = [pair for pair in needs_translation if pair not in by_key]
    if missing:
        logger.warning("facet-translator: %d of %d value(s) came back untranslated (degraded)",
                       len(missing), len(needs_translation))
    return out, bool(missing)
