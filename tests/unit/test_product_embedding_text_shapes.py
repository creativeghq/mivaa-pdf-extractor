from app.api.pdf_processing.stage_4_products import build_product_embedding_text


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
