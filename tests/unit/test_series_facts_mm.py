import pytest

sf = pytest.importorskip("app.services.discovery.series_facts")

HEADERS = ["Part", "Colour", "Size (mm)", "Pieces per pack", "Piece weight (kg)"]
TABLE = {"table_data": {"headers": HEADERS, "rows": [
    ["Panel", "Teak", "600x600x9", 1, None],
    ["Panel", "Oak", "600x600x9", 1, None],
    ["Panel", "Teak", "2780x600x9", 1, 7.1],
]}}
LEAKED = ["2780×600×9 mm", "2780×600×20 mm", "2800×1200×1200 mm"]


def _product(**metadata):
    return {"name": "Comfort 9", "description": "A panel 9 mm thick.", "metadata": metadata, "attributes": {}}


def test_sizes_come_from_the_column_headed_in_millimetres():
    assert sf.table_mm_sizes([TABLE]) == ["600×600×9 mm", "2780×600×9 mm"]


def test_headers_held_beside_the_rows_count_too():
    data = TABLE["table_data"]
    assert sf.table_mm_sizes([{"headers": data["headers"], "table_data": {"rows": data["rows"]}}]) == ["600×600×9 mm", "2780×600×9 mm"]


def test_a_table_with_no_millimetre_size_column_says_nothing():
    other = {"table_data": {"headers": ["Model", "Weight (kg)"], "rows": [["A", "600x600x9"]]}}
    assert sf.table_mm_sizes([other]) == []


def test_settling_drops_another_models_sizes_and_adds_what_the_table_has():
    out = sf.settle_product(_product(available_sizes=LEAKED, dimensions=["2780x600x9"]), [TABLE])
    assert out["metadata"]["available_sizes"] == ["600×600×9 mm", "2780×600×9 mm"]
    assert out["metadata"]["available_formats"] == ["600×600×9 mm", "2780×600×9 mm"]
    assert "dimensions" not in out["metadata"]
    assert out["changes"]["sizes"]["dropped"] == ["2780×600×20 mm", "2800×1200×1200 mm"]
    assert out["description"] == "A panel 9 mm thick."


def test_settling_millimetre_sizes_twice_changes_nothing():
    first = sf.settle_product(_product(available_sizes=LEAKED), [TABLE])
    product = {**_product(), "metadata": {k: v for k, v in first["metadata"].items() if k != "series_facts"}}
    assert sf.settle_product(product, [TABLE])["changes"] == {}


def test_the_audit_names_wrong_and_missing_millimetre_sizes():
    assert "sizes_unverified" in sf.fact_problems(_product(available_sizes=LEAKED), [TABLE])
    assert "sizes_unverified" in sf.fact_problems(_product(), [TABLE])
    assert "sizes_unverified" not in sf.fact_problems(_product(available_sizes=["600×600×9 mm", "2780×600×9 mm"]), [TABLE])


def test_a_centimetre_product_is_left_to_the_centimetre_rules():
    packing = {"table_data": {"headers": ["Format", "Pieces", "kg"], "rows": [["60x120 cm", "2", "30"]]}}
    out = sf.settle_product(_product(available_sizes=["60x120 cm", "30x60 cm"]), [packing])
    assert out["metadata"]["available_formats"] == ["60x120"]
