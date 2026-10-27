import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

_SRC = Path(__file__).resolve().parents[2] / "app" / "api" / "pdf_processing" / "stage_4_products.py"
_tree = ast.parse(_SRC.read_text(encoding="utf-8"))
_fn = next(n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_product_embedding_text")
_ns: Dict[str, Any] = {"Any": Any, "Dict": Dict, "List": List, "Optional": Optional}
exec(compile(ast.Module(body=[_fn], type_ignores=[]), str(_SRC), "exec"), _ns)
build_product_embedding_text = _ns["build_product_embedding_text"]


def _text(metadata, fields):
    return build_product_embedding_text("Nova", "A tile", metadata, fields)


def test_dict_valued_spec_reaches_the_embedding_text():
    out = _text({"thickness": {"value": 9, "unit": "mm"}}, ["thickness"])
    assert "thickness: value 9, unit mm" in out


def test_nested_non_scalars_in_a_dict_spec_are_skipped():
    out = _text({"thickness": {"unit": "mm", "range": [1, 2]}}, ["thickness"])
    assert "thickness: unit mm" in out and "range" not in out


def test_scalar_and_list_specs_render_as_before():
    out = _text({"finish": "matt", "colors": ["grey", "white"]}, ["finish", "colors"])
    assert out == "Nova | A tile | finish: matt | colors: grey, white"
