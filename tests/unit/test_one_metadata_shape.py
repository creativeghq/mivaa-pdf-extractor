"""
`products.metadata` has ONE shape (#347 phase 2.1).

The extractor returns metadata grouped into sections — `material_properties`, `dimensions`,
`appearance`, `electrical_specs`, `performance`, `packaging`. Those sections are a PROMPT device:
they tell the model what to hunt for, per category, and the field registry already records which
section a field belongs to. They were never a storage schema.

But `product_discovery_service` had three enrichment paths and they disagreed:

  * the validated path ran the prototype validator's `_flatten_metadata` -> FLAT
  * the validation-failure fallback flattened by hand                    -> FLAT
  * the no-validation path spread `discovered` straight into the merge   -> NESTED

Which shape a document ended up with depended on which branch happened to run.

That is not cosmetic, and it is the reason this test exists. `FacetCanonicalizer._collect_pending`
and `collect_raw_attributes` iterate TOP-LEVEL keys only. So for any product enriched down the
nested path, `finish` — a whitelisted, canonicalizable facet — never reached
`products.attributes`. It was extracted, stored, rendered on the product page, and silently
unfilterable. Nothing raised, nothing logged, and the value was right there in the JSON.

It also reaches money: `product_m2_per_piece()` reads `products.metadata.dimensions`, so a
mis-shaped row makes `convert_to_base_unit` return NULL, and a configured pallet price break
"simply never matches" — silently.
"""
import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load_leaf(rel: str, name: str):
    """Load a leaf module BY PATH, so no `app` package __init__ runs and no DB client is built.
    Same reason `test_no_fallback_embedder` is source-based: these must run in CI in a second."""
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


flatten_extracted_metadata = _load_leaf(
    "app/services/metadata/metadata_shape.py", "metadata_shape"
).flatten_extracted_metadata


def test_sections_are_flattened_away():
    """The tiles shape: sections in, plain keys out."""
    out = flatten_extracted_metadata({
        "discovered": {
            "material_properties": {"finish": "matt", "thickness_mm": 9},
            "appearance": {"colors": ["white", "beige"]},
            "dimensions": {"available_sizes": ["60x60 cm", "30x60 cm"]},
        },
        "critical": {"material_category": "tiles"},
    })
    assert out == {
        "finish": "matt",
        "thickness_mm": 9,
        "colors": ["white", "beige"],
        "available_sizes": ["60x60 cm", "30x60 cm"],
        "material_category": "tiles",
    }
    # The section names must not survive as keys — that was the whole bug.
    assert "material_properties" not in out
    assert "appearance" not in out


def test_whitelisted_facets_land_top_level_where_the_collector_looks():
    """
    The specific failure: a canonicalizable facet nested one level deep is invisible.

    `collect_raw_attributes` / `_collect_pending` decide what becomes a filterable attribute by
    walking TOP-LEVEL keys and asking `is_canonicalizable(key)`. Under the nested shape the only
    top-level keys are SECTION names, and no section name is a facet — so the whole product
    canonicalises to nothing. Checked against the real whitelist rather than a restatement of it.
    """
    is_canonicalizable = _load_leaf(
        "app/services/facets/facet_whitelist.py", "facet_whitelist_probe"
    ).is_canonicalizable

    extracted = {
        "discovered": {"material_properties": {"finish": "Matt Black"}},
        "critical": {},
    }

    # The old nested shape — what the no-validation path produced. Its top-level key is the
    # section, and a section is not a facet, so nothing is collectable.
    nested = dict(extracted["discovered"])
    assert list(nested) == ["material_properties"]
    assert not any(is_canonicalizable(k) for k in nested), (
        "sanity check: under the nested shape the collector sees only section names"
    )

    # The one shape puts the facet where the collector actually looks.
    flat = flatten_extracted_metadata(extracted)
    assert is_canonicalizable("finish")
    assert [k for k in flat if is_canonicalizable(k)] == ["finish"]


def test_critical_beats_discovered():
    """Precedence matches every caller: discovered < critical."""
    out = flatten_extracted_metadata({
        "discovered": {"material_properties": {"material_category": "guessed"}},
        "critical": {"material_category": "tiles"},
    })
    assert out["material_category"] == "tiles"


def test_a_value_emitted_outside_any_section_is_kept():
    """The hand-rolled fallback dropped these with an isinstance check and no log line."""
    out = flatten_extracted_metadata({
        "discovered": {"factory_name": "Acme SA", "material_properties": {"finish": "gloss"}},
        "critical": {},
    })
    assert out["factory_name"] == "Acme SA"
    assert out["finish"] == "gloss"


@pytest.mark.parametrize("extracted", [
    {},
    {"discovered": None, "critical": None},
    {"discovered": {}, "critical": {}},
    {"discovered": {"material_properties": None}},
])
def test_empty_and_null_inputs_do_not_raise(extracted):
    """Extraction failures hand us partial dicts; shaping them must never be the thing that throws."""
    assert flatten_extracted_metadata(extracted) == {} or isinstance(
        flatten_extracted_metadata(extracted), dict
    )


def test_all_three_enrichment_paths_use_the_one_flattener():
    """
    Guard the guard. A future edit that re-inlines a flatten — or spreads `discovered` again —
    reintroduces exactly the divergence this phase removed, and no runtime test would catch it
    because both shapes are valid JSON.
    """
    src = _ROOT / "app" / "services" / "discovery" / "product_discovery_service.py"
    text = src.read_text(encoding="utf-8")

    assert text.count("flatten_extracted_metadata(extracted)") >= 2, (
        "both enrichment paths must shape metadata through the shared flattener"
    )
    assert "**extracted.get(\"discovered\", {})" not in text, (
        "spreading `discovered` directly is the nested-shape bug (#347 defect 1)"
    )
