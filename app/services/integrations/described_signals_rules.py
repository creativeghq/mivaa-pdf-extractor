"""Described-signal rules (#499): the schemas, spec cleaning and the evidence check. Import-free."""

from __future__ import annotations

import re
from typing import Any, Dict, List

_WS = re.compile(r"\s+")

SOURCES: Dict[str, str] = {
    "job_posting": "job ads companies publish on their own job boards",
    "technology": "technologies detected on company websites and in their job ads",
    "page_change": "meaningful changes on web pages this workspace monitors",
    "mention": "mentions of this workspace's brand online",
    "price_move": "unusual price moves on products this workspace tracks",
}


COMPILE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string"},
        "sources": {"type": "array", "items": {"type": "string", "enum": list(SOURCES)}},
        "keywords": {"type": "array", "items": {"type": "string"}},
        "must_not": {"type": "array", "items": {"type": "string"}},
        "judge_question": {"type": "string"},
        "positive_examples": {"type": "array", "items": {"type": "string"}},
        "negative_examples": {"type": "array", "items": {"type": "string"}},
        "min_similarity": {"type": "number"},
    },
    "required": ["name", "sources", "keywords", "must_not", "judge_question",
                 "positive_examples", "negative_examples", "min_similarity"],
}

JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "match": {"type": "boolean"},
        "evidence_quote": {"type": "string"},
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["match", "evidence_quote", "reason", "confidence"],
}


def _clean_list(values: Any, max_items: int, max_len: int) -> List[str]:
    out: List[str] = []
    for v in values if isinstance(values, list) else []:
        s = str(v).strip()[:max_len]
        if s and s.lower() not in {x.lower() for x in out}:
            out.append(s)
        if len(out) >= max_items:
            break
    return out


def clean_spec(raw: Dict[str, Any]) -> Dict[str, Any]:
    """The schema cannot carry ranges (the API drops them), so lengths and bounds are enforced here."""
    sources = [s for s in _clean_list(raw.get("sources"), 5, 40) if s in SOURCES]
    question = str(raw.get("judge_question") or "").strip()[:600]
    if not sources or not question:
        raise ValueError("the drafted rules named no source or no judge question")
    try:
        min_sim = float(raw.get("min_similarity"))
    except (TypeError, ValueError):
        min_sim = 0.45
    return {
        "sources": sources,
        "keywords": _clean_list(raw.get("keywords"), 40, 80),
        "must_not": _clean_list(raw.get("must_not"), 20, 80),
        "judge_question": question,
        "positive_examples": _clean_list(raw.get("positive_examples"), 5, 300),
        "negative_examples": _clean_list(raw.get("negative_examples"), 5, 300),
        "min_similarity": max(0.2, min(0.9, min_sim)),
    }


def quote_is_in_item(quote: str, item_text: str) -> bool:
    """A match must quote the item verbatim (whitespace and case aside) — the quote IS the evidence."""
    q = _WS.sub(" ", (quote or "").strip()).lower()
    return len(q) >= 8 and q in _WS.sub(" ", item_text or "").lower()
