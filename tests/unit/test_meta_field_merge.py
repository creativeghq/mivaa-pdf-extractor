"""
The 3-source meta merge must be TOTAL (#347 phase 2.1 follow-up).

`_build_enriched_product_data` folds keyword-scanned chunk values into whatever the AI
extraction already found — the "belt and suspenders" described in docs/meta-field-aggregation.md,
where "NOVA available in Clay" on page 31 still reaches a product created from page 12.

Why this file exists: that merge raised on two shapes that occur in real data.

  * `colors` as a list of ``{"name": ..., "hex": ...}`` objects — a shape the frontend's
    `getAvailableColors` explicitly supports — built a ``set()`` over dicts and threw
    ``unhashable type: 'dict'``.
  * any list mixing strings with a number threw in ``sorted()``, which cannot order ``int``
    against ``str``.

And the failure is worse than a crash. `_build_enriched_product_data` catches everything and
falls back to `_create_product_from_chunk`, so the product is still created — silently stripped
of every attribute the enrichment found, leaving one log line nobody reads. A valid-looking row
with no metadata is exactly the silent-zero shape this codebase keeps getting bitten by.

Both shapes became REACHABLE when phase 2.1 flattened `products.metadata`: `colors` used to sit
nested under `appearance`, so `metadata['colors']` was absent and the merge took its
"nothing here yet" branch. Flat keys mean the real branches now run.

Loaded by AST rather than imported: `product_creation_service` pulls in the Supabase client, and
MIVAA's unit tests import no app module — see `test_no_fallback_embedder`.
"""
import ast
from pathlib import Path

import pytest

_SRC = (
    Path(__file__).resolve().parents[2]
    / "app" / "services" / "products" / "product_creation_service.py"
)


def _load_merge():
    """Compile just `_merge_meta_field` out of the module, with no imports executed."""
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_merge_meta_field"
    )
    fn.decorator_list = []  # drop @staticmethod so it is a plain function
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Any": object, "List": list}
    exec(compile(module, "<merge>", "exec"), namespace)  # noqa: S102 — our own source
    return namespace["_merge_meta_field"]


merge = _load_merge()


def test_the_normal_case_is_unchanged():
    """Case-insensitive union of two string lists — the documented behaviour, byte for byte."""
    assert merge(["White", "Beige"], ["clay", "natural"]) == [
        "beige", "clay", "natural", "white",
    ]


def test_a_single_string_becomes_a_merged_list():
    assert merge("White", ["clay"]) == ["clay", "white"]


def test_absent_or_empty_existing_takes_the_aggregated_values():
    for empty in (None, "", [], {}):
        assert merge(empty, ["clay", "White"]) == ["clay", "white"]


@pytest.mark.parametrize("existing", [
    [{"name": "White", "hex": "#fff"}],       # the shape getAvailableColors supports
    ["White", 3],                              # mixed list — sorted() could not order these
    {"primary": "white"},                      # a dict value
    [{"a": 1}, {"b": 2}],
])
def test_unfoldable_existing_values_are_kept_verbatim_and_never_raise(existing):
    """
    The original code already stated the rule for the top-level dict case — *AI extraction takes
    priority*. It simply never applied it to list ELEMENTS. Structured colour objects carry more
    than the keyword scanner's strings, so keeping them is the correct resolution, not a
    fallback.
    """
    before = repr(existing)
    assert merge(existing, ["clay"]) == existing
    assert repr(existing) == before, "must not mutate the caller's value"


def test_junk_from_the_aggregator_is_ignored_rather_than_fatal():
    assert merge(["White"], [None, "", "  ", "Clay"]) == ["clay", "white"]


def test_no_aggregated_values_leaves_existing_untouched():
    assert merge(["White"], []) == ["White"]
    assert merge(["White"], [None]) == ["White"]


def test_the_merge_is_total_over_every_shape_seen_in_practice():
    """
    The contract in one assertion: nothing this function is handed may raise. Its caller's
    except-branch silently downgrades the product, so a raise here is invisible in production.
    """
    values = [
        None, "", "White", ["White"], ["White", "Beige"], [], {},
        {"primary": "white"}, [{"name": "White"}], ["White", 3], [3, 4], 7, 1.5, True,
    ]
    for existing in values:
        for aggregated in ([], ["clay"], [None, "clay"], ["Clay", "clay"]):
            merge(existing, aggregated)  # must simply not raise


def test_the_raising_implementation_is_gone():
    """Guard the guard: the exact expressions that threw must not come back."""
    text = _SRC.read_text(encoding="utf-8")
    assert "set(v.lower() if isinstance(v, str) else v for v in metadata[field_type])" not in text
    assert "_merge_meta_field" in text
