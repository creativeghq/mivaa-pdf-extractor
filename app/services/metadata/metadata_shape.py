"""
The ONE shape `products.metadata` takes (#347 phase 2.1).

Deliberately a leaf module: it imports nothing from `app`, so a guard test can load it by path
without pulling in the Supabase client. MIVAA's unit tests are source-based for exactly that
reason — see `tests/unit/test_no_fallback_embedder.py`.
"""

from typing import Any, Dict


def flatten_extracted_metadata(extracted: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse an extractor result into the FLAT shape `products.metadata` takes.

    The extractor returns ``{"critical": {...}, "discovered": {section: {field: value}},
    "unknown_attributes": {...}}``. Sections (``material_properties``, ``dimensions``,
    ``appearance``, ``electrical_specs``, ...) are a PROMPT device — they tell the model what
    to look for. They are not a storage schema, and the registry already knows which section a
    field belongs to, so nothing downstream needs them nested.

    Why this function exists: there were THREE enrichment paths in
    ``product_discovery_service`` and they did not agree. The validated path ran
    ``_flatten_metadata`` and produced flat keys; the validation-failure fallback flattened by
    hand; and the no-validation path spread ``discovered`` straight into the merge, leaving
    ``{"material_properties": {"finish": "matt"}}`` nested. Which shape a document ended up
    with depended on which path happened to run.

    That was not cosmetic. ``FacetCanonicalizer._collect_pending`` and
    ``collect_raw_attributes`` iterate TOP-LEVEL keys only, so ``finish`` — a whitelisted,
    canonicalizable facet — never reached ``products.attributes`` for any product that went
    down the nested path. It was extracted, stored, displayed, and silently unfilterable.

    Non-dict section values are KEPT rather than dropped: a model that emits a field outside
    any section is giving us data, and the old hand-rolled fallback discarded it with an
    ``isinstance`` check and no log line.

    Precedence matches the callers: discovered < critical.
    """
    flat: Dict[str, Any] = {}
    for section, fields in (extracted.get("discovered") or {}).items():
        if isinstance(fields, dict):
            flat.update(fields)
        elif fields is not None:
            # Emitted outside any section — a value, not a section. Keep it.
            flat[section] = fields
    flat.update(extracted.get("critical") or {})
    return flat
