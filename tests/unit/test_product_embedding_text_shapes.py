import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

_SRC = Path(__file__).resolve().parents[2] / "app" / "api" / "pdf_processing" / "stage_4_products.py"
_tree = ast.parse(_SRC.read_text(encoding="utf-8"))
_WANTED = ("build_product_embedding_text", "embedding_inputs_changed")
_fns = [n for n in _tree.body if isinstance(n, ast.FunctionDef) and n.name in _WANTED]
_ns: Dict[str, Any] = {"Any": Any, "Dict": Dict, "List": List, "Optional": Optional}
exec(compile(ast.Module(body=_fns, type_ignores=[]), str(_SRC), "exec"), _ns)
build_product_embedding_text = _ns["build_product_embedding_text"]
embedding_inputs_changed = _ns["embedding_inputs_changed"]
_SOURCE = _SRC.read_text(encoding="utf-8")


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


def _changed(old_desc, old_meta, new_desc, new_meta, fields=("finish", "weight_kg")):
    return embedding_inputs_changed("Nova", old_desc, old_meta, new_desc, new_meta, list(fields))


def test_a_filled_spec_field_makes_the_vector_stale_even_without_a_new_description():
    assert _changed("A tile", {"finish": "matt"}, "A tile", {"finish": "matt", "weight_kg": 12.5})


def test_a_new_description_makes_the_vector_stale():
    assert _changed(None, {"finish": "matt"}, "A matt tile", {"finish": "matt"})


def test_a_field_the_embedding_text_never_reads_does_not_cause_a_re_embed():
    assert not _changed("A tile", {"finish": "matt"}, "A tile", {"finish": "matt", "internal_note": "x"})


def test_nothing_changed_means_no_re_embed():
    assert not _changed("A tile", {"finish": "matt"}, "A tile", {"finish": "matt"})


def test_enrichment_re_embeds_on_changed_inputs_and_keeps_the_existing_description():
    call = _SOURCE[_SOURCE.index("final_description = new_description or existing_description"):]
    call = call[:call.index("products_reembedded")]
    assert "embedding_inputs_changed(" in call
    assert "reembed_product_text(" in call and "final_description, new_metadata" in call
    assert "if new_description:" not in call
