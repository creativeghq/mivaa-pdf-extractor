"""
Guard for the VLM table path: layout cache content → product_tables → product metadata.

This whole branch was dead from #248 (2026-07-04) to 2026-08-02: PaddleOCR-VL recognized
every table to markdown and cached it on the layout element, nothing parsed it,
`product_tables` was never written, and `TableMetadataExtractor` mined an always-empty
table on every product — the `ops.silent_zero` shape, invisible to typecheck and to every
integrity probe because zero rows is a valid row count.

Turning it on exposed three latent bugs in the consumer, all of which produce a *wrong
number that is still a valid number*:
  1. `_find_column` matched keywords as substrings, so the single-letter height keyword
     'l' hit `Pz/Scatola` and a 60x120 tile was recorded as 60x2.
  2. Packaging keyword sets had no Italian/Spanish abbreviations ('Pz', 'Mq'), so
     pieces-per-box and coverage never populated on the catalogs we actually ingest.
  3. Spec tables are Property | Standard | Value; taking row[1] recorded 'ISO 10545-3'
     as the water-absorption value.

Each is pinned below. Loaded by path so the guard runs in CI with nothing installed but
pytest — importing `app.services.*` as a package would pull in the Supabase client and
the entire runtime dependency set.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load(module_name: str, relative_path: str):
    path = _ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


table_extraction = _load("table_extraction_under_test", "app/services/pdf/table_extraction.py")

# table_metadata_extractor imports the shared keyword sets from table_extraction. Register
# the already-loaded module under its real name so that import resolves without executing
# app/services/__init__.py.
for _pkg in ("app", "app.services", "app.services.pdf", "app.services.metadata"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
sys.modules["app.services.pdf.table_extraction"] = table_extraction

table_metadata_extractor = _load(
    "table_metadata_extractor_under_test",
    "app/services/metadata/table_metadata_extractor.py",
)

TableExtractor = table_extraction.TableExtractor
TableMetadataExtractor = table_metadata_extractor.TableMetadataExtractor
parse_table_content = table_extraction.parse_table_content

# The canonical Italian catalog table: dimensions AND packaging in one grid.
CATALOG_MARKDOWN = """| Formato | Spessore | Pz/Scatola | Mq/Scatola | Kg/Scatola | Scatole/Pallet |
|---------|:--------:|-----------:|------------|------------|----------------|
| 60x120 cm | 9 mm | 2 | 1,44 | 31,5 | 30 |
| 30x60 cm | 9 mm | 6 | 1,08 | 23,8 | 40 |
"""

# The canonical spec table: three columns, the measurement in the last one.
SPEC_HTML = (
    "<table>"
    "<tr><th>Caratteristica</th><th>Norma</th><th>Valore</th></tr>"
    "<tr><td>Assorbimento <b>acqua</b></td><td>ISO 10545-3</td><td>&lt; 0,5%</td></tr>"
    "<tr><td>Resistenza allo scivolamento</td><td>DIN 51130</td><td>R10</td></tr>"
    "<tr><td>Resistenza al gelo</td><td>ISO 10545-12</td><td>Conforme</td></tr>"
    "</table>"
)


@pytest.fixture
def extractor():
    # No Supabase client: every method exercised here is pure parsing.
    return TableMetadataExtractor(None)


# ---------------------------------------------------------------- content parsing


def test_markdown_table_parses_and_drops_the_alignment_row():
    grid = parse_table_content(CATALOG_MARKDOWN)
    assert grid is not None
    assert grid[0][0] == "Formato"
    # 1 header + 2 data rows — the |---|:--:| row is syntax, not data.
    assert len(grid) == 3
    assert grid[1] == ["60x120 cm", "9 mm", "2", "1,44", "31,5", "30"]


def test_html_table_parses_with_nested_tags_and_entities():
    grid = parse_table_content(SPEC_HTML)
    assert grid is not None
    assert len(grid) == 4
    # Nested <b> must not split the cell; &lt; must be unescaped.
    assert grid[1][0] == "Assorbimento acqua"
    assert grid[1][2] == "< 0,5%"


@pytest.mark.parametrize(
    "content",
    ["", "   ", "just some prose with no table", "| only one row |", "| a |\n| b |"],
)
def test_unusable_content_returns_none_not_an_empty_table(content):
    # Callers skip the region on None. Persisting an empty table would look like
    # a successful extraction and mask the failure.
    assert parse_table_content(content) is None


# ---------------------------------------------------------------- classification


def test_combined_catalog_table_classifies_as_dimensions():
    grid = parse_table_content(CATALOG_MARKDOWN)
    ex = TableExtractor()
    record = ex._convert_table_to_dict(grid, page_number=13, confidence=0.9)
    assert ex.classify_table_type(record["headers"], record["table_data"]) == "dimensions"
    assert record["extractor"] == "paddleocr-vl"


def test_italian_spec_table_classifies_as_specifications():
    grid = parse_table_content(SPEC_HTML)
    ex = TableExtractor()
    record = ex._convert_table_to_dict(grid, page_number=14, confidence=0.9)
    assert ex.classify_table_type(record["headers"], record["table_data"]) == "specifications"


# ---------------------------------------------------------------- the three number bugs


def test_size_column_wins_over_a_spuriously_matched_axis_column(extractor):
    """Bug 1: 'l' matched `Pz/Scatola`, recording 60x120 as 60x2."""
    grid = parse_table_content(CATALOG_MARKDOWN)
    headers, rows = grid[0], grid[1:]
    dims = extractor._parse_dimensions_table(headers, {"headers": headers, "rows": rows})

    assert [(d["width"], d["height"]) for d in dims] == [(60.0, 120.0), (30.0, 60.0)]
    assert all(d["thickness"] == 9.0 for d in dims)
    assert extractor._dimensions_to_sizes(dims) == ["30x60 cm", "60x120 cm"]


def test_italian_packaging_abbreviations_populate(extractor):
    """Bug 2: 'Pz' and 'Mq' were absent, so these fields never populated."""
    grid = parse_table_content(CATALOG_MARKDOWN)
    headers, rows = grid[0], grid[1:]
    pkg = extractor._parse_packaging_table(headers, {"headers": headers, "rows": rows})

    assert pkg == {
        "pieces_per_box": 2.0,
        "boxes_per_pallet": 30.0,
        "weight_per_box_kg": 31.5,
        "coverage_per_box_m2": 1.44,
    }


def test_spec_value_is_the_measurement_not_the_standard(extractor):
    """Bug 3: row[1] recorded 'ISO 10545-3' as the water-absorption value."""
    grid = parse_table_content(SPEC_HTML)
    headers, rows = grid[0], grid[1:]
    specs = extractor._parse_specifications_table(headers, {"headers": headers, "rows": rows})

    assert specs["performance"]["water_absorption"] == "< 0,5%"
    assert specs["performance"]["slip_resistance"] == "R10"
    assert specs["performance"]["frost_resistance"] == "Conforme"


def test_split_axis_english_table_still_works(extractor):
    """The axis-column fallback must survive the size-column precedence fix."""
    headers = ["Width", "Height", "Thickness", "Pcs/Box"]
    rows = [["30", "60", "8", "10"]]
    dims = extractor._parse_dimensions_table(headers, {"headers": headers, "rows": rows})

    assert dims == [{"width": 30.0, "height": 60.0, "thickness": 8.0, "unit": "cm"}]
    pkg = extractor._parse_packaging_table(headers, {"headers": headers, "rows": rows})
    assert pkg["pieces_per_box"] == 10.0


# ------------------------------------------------- dimensions AND packaging together


def test_one_table_yields_both_dimensions_and_packaging(extractor):
    """The elif chain this replaces let a single label discard the other half.

    The canonical catalog table carries both, so a 'dimensions' label used to mean
    packaging was never parsed from the only table that had it.
    """
    grid = parse_table_content(CATALOG_MARKDOWN)
    headers = grid[0]
    assert extractor._looks_like_dimensions(headers)
    assert extractor._looks_like_packaging(headers, {"headers": headers, "rows": grid[1:]})
