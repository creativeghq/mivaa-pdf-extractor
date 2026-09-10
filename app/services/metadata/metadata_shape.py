"""
The ONE shape `products.metadata` takes (#347 phase 2.1).

Deliberately a leaf module: it imports nothing from `app`, so a guard test can load it by path
without pulling in the Supabase client. MIVAA's unit tests are source-based for exactly that
reason — see `tests/unit/test_no_fallback_embedder.py`.
"""

from typing import Any, Dict


def flatten_extracted_metadata(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse an extractor result into the FLAT shape `products.metadata` takes."""
    flat: Dict[str, Any] = {}
    for section, fields in (extracted.get("discovered") or {}).items():
        if isinstance(fields, dict):
            flat.update(fields)
        elif fields is not None:
            # Emitted outside any section — a value, not a section. Keep it.
            flat[section] = fields
    flat.update(extracted.get("critical") or {})
    return flat
