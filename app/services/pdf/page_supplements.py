"""Chunks for printed numbers the layout pass dropped: the exact text-layer lines that carry them."""

from typing import Any, Dict, List, Optional, Sequence

SOURCE = "text_layer_supplement"
CONTEXT_LINES = 1


def supplement_text(lines: Sequence[Any], missing: Sequence[str], numeric_tokens: Any) -> List[str]:
    wanted = set(missing)
    keep: set = set()
    for i, line in enumerate(lines):
        if numeric_tokens(line.text()) & wanted:
            keep.update(range(max(0, i - CONTEXT_LINES), min(len(lines), i + CONTEXT_LINES + 1)))
    return [lines[i].text() for i in sorted(keep)]


def owner_for(page: int, products: Sequence[Dict[str, Any]]) -> Optional[str]:
    owners = [p["id"] for p in products if page in ((p.get("metadata") or {}).get("page_range") or [])]
    return owners[0] if len(owners) == 1 else None


def build_supplement(label: str, page: int, text_lines: Sequence[str], missing: Sequence[str]) -> Optional[Dict[str, Any]]:
    if not text_lines:
        return None
    return {
        "content": f"{label} - page {page} (printed text)\n" + "\n".join(text_lines),
        "metadata": {"source": SOURCE, "page_number": page, "missing_numbers": sorted(missing)[:80], "is_table": False},
    }
