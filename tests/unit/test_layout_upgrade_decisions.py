import ast
import importlib.util
import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
SOURCE = (_ROOT / "app/services/pdf/layout_upgrade.py").read_text(encoding="utf-8")


def _pure():
    tree = ast.parse(SOURCE)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"stale_pages", "decide"}]
    import typing

    namespace = {"ROUTE_TEXT_LAYER": "text_layer", "Any": typing.Any, "Dict": typing.Dict, "List": typing.List, "Optional": typing.Optional}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "layout_upgrade", "exec"), namespace)
    return namespace


def test_only_rows_without_a_route_are_stale():
    ns = _pure()
    rows = [{"page_number": 1, "analysis_metadata": {}}, {"page_number": 2, "analysis_metadata": {"route": "ocr"}}, {"page_number": 3, "analysis_metadata": None}]
    assert [r["page_number"] for r in ns["stale_pages"](rows)] == [1, 3]


def test_a_truncated_ocr_table_is_replaced_only_when_the_text_layer_can_stand_in():
    decide = _pure()["decide"]
    planned = [{"region_type": "TABLE"}]
    assert decide("text_layer", {"truncated": 1}, planned) == "replace_with_text_layer"
    assert decide("text_layer", {"truncated": 0}, planned) == "annotate"
    assert decide("ocr_with_reference", {"truncated": 2}, None) == "annotate"
    assert decide("text_layer", {"truncated": 1}, None) == "annotate"


def test_an_upgrade_never_rewrites_product_tables_and_never_calls_ocr():
    code = SOURCE
    assert "persist_tables_from_layout_cache" not in code and "run_structural_pass" not in code
    assert "product_tables" not in code
    assert "apply" in code and "if apply:" in code
