"""Gemini text model ids. Import-free twin of `supabase/functions/_shared/gemini-models.ts`."""

from __future__ import annotations

from typing import Any, Dict, Optional

GEMINI_TEXT = "gemini-4-hydrogen"
GEMINI_FRONTIER = "gemini-4-carbon"


def gemini_thinking_config(model: str, level: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Gemini 4 thinks at `high` unless told otherwise and counts thoughts against
    `maxOutputTokens`; `low` is its floor (`minimal` and `thinkingBudget` are a 400)."""
    if model.startswith("gemini-4"):
        return {"thinkingLevel": level if level and level != "none" else "low"}
    return {"thinkingLevel": level} if level and level != "none" else None


def gemini_output_tokens(usage_metadata: Any) -> int:
    """Output tokens as billed: the answer plus the thinking Gemini reports separately."""
    u = usage_metadata or {}
    return int(u.get("candidatesTokenCount") or 0) + int(u.get("thoughtsTokenCount") or 0)
