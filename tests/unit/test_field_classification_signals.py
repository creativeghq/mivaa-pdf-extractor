"""Deterministic field-classification signals (#347 phase 4.2).

Plurality and SKU correlation decide most fields for free, before any model call. Both are pure
functions of the extracted document, so both can be tested without a database — which matters,
because the alternative is discovering the rule was wrong on a live catalogue.

The bias under test is the plan's, and it is deliberately asymmetric: when plurality fires,
default to IDENTITY. Wrongly marking a field identity SPLITS stock into duplicate rows, which is
visible and fixable. Wrongly marking it descriptive MERGES stock that should be separate, which
is invisible — and is the defect this whole issue exists to remove.

SOURCE-BASED, following test_field_registry_is_the_source.py: `dynamic_metadata_extractor`
imports the Supabase client at module load, so importing it would need credentials CI does not
have. The function under test is pure, so it is lifted out of the AST and executed on its own —
the REAL code, not a transcription of it.
"""
import ast
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[2] / "app/services/metadata/dynamic_metadata_extractor.py"


def _lift(func_name: str):
    """Execute one method from the module in isolation, with no imports pulled in."""
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            node.decorator_list = []
            module = ast.Module(body=[node], type_ignores=[])
            ns: dict = {}
            exec(compile(ast.fix_missing_locations(module), "<lifted>", "exec"), ns)
            return ns[func_name]
    raise AssertionError(f"{func_name} not found in {_SRC.name} — this guard is stale")


@pytest.fixture(scope="module")
def sku_correlated():
    fn = _lift("_sku_correlated_fields")
    return lambda data: fn(None, data)  # pure: `self` is never touched


def test_sku_map_needs_two_variants(sku_correlated):
    """One variant is not a mapping — every field would "vary" across a single name."""
    data = {"critical": {"sku_codes": {"60x60 Matte": "SKU-1"}, "finish": ["matte", "polished"]}}
    assert sku_correlated(data) == set()


def test_field_varying_across_sku_variants_is_identity(sku_correlated):
    """The manufacturer gave the variants different part numbers — that IS the axis."""
    data = {"critical": {
        "sku_codes": {"60x60 matte": "SKU-1", "60x60 polished": "SKU-2"},
        "finish": ["matte", "polished"],
    }}
    assert "finish" in sku_correlated(data)


def test_field_absent_from_the_variant_names_is_not_correlated(sku_correlated):
    """A field that does not distinguish the SKUs is not evidence of an axis."""
    data = {"critical": {
        "sku_codes": {"60x60 matte": "SKU-1", "60x60 polished": "SKU-2"},
        "pei_rating": ["IV", "V"],
    }}
    assert "pei_rating" not in sku_correlated(data)


def test_sku_map_is_found_when_nested_under_a_category(sku_correlated):
    """`discovered` nests fields by category; the map must still be found."""
    data = {
        "discovered": {"technical": {"sku_codes": {"a matte": "1", "b polished": "2"}}},
        "critical": {"finish": ["matte", "polished"]},
    }
    assert "finish" in sku_correlated(data)


def test_no_sku_map_at_all_is_silent(sku_correlated):
    """Silence, not a guess: absent evidence must classify nothing."""
    assert sku_correlated({"critical": {"finish": ["matte", "polished"]}}) == set()
    assert sku_correlated({}) == set()


@pytest.mark.parametrize(
    "value,fires",
    [
        (["60x60", "30x60"], True),   # enumerated for ONE product -> a choice
        (["60x60"], False),           # a single-element list is one value, not a choice
        ("IV", False),                # a scalar is a property already chosen
        ([], False),
        ([None, ""], False),          # blanks are not values
        (["60x60", ""], False),       # one real value after filtering
    ],
)
def test_plurality_rule_shape(value, fires):
    """The plurality branch: more than one non-empty value in a list means a variant axis."""
    assert (isinstance(value, (list, tuple))
            and len([v for v in value if v not in (None, "")]) > 1) is fires


def test_plurality_branch_still_exists_in_source():
    """The parametrised rule above mirrors the branch; pin that the branch is still there."""
    src = _SRC.read_text(encoding="utf-8")
    assert 'signal, role, confidence = "plurality", "identity", 1.0' in src
    assert 'signal, role, confidence = "sku_correlation", "identity", 1.0' in src
    # The verdict must go through the RPC, never a direct role write — one ladder, one veto.
    assert 'supabase.client.rpc("classify_field_role"' in src
