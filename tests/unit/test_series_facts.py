"""One product states one set of sizes and colours; every case here was a real Keros series."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for _pkg in ("app", "app.services", "app.services.discovery"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
sys.modules["app.services.discovery"].series_reconcile = _load("app.services.discovery.series_reconcile", "app/services/discovery/series_reconcile.py")
sf = _load("series_facts_under_test", "app/services/discovery/series_facts.py")

HEADERS = ["Format", "Variant", "Pieces / box", "m2 / box", "kg / box", "Pallet (cm)", "Boxes / pallet", "m2 / pallet", "kg / pallet",
           "Europallet boxes", "Europallet m2", "Europallet kg"]


def packing(*rows):
    return {"table_data": {"headers": HEADERS, "rows": [list(r) for r in rows]}}


GOOD_ROW = ["30X60", "(K)", "7", "1,26", "22,68", "103X103", "48", "60,48", "1100", "40", "50,40", "930"]


def test_a_size_nothing_supports_goes_and_a_decimal_is_not_cut_in_half():
    product = {"name": "Abelia", "description": "", "attributes": {},
               "metadata": {"available_formats": ["18.2x122"], "available_sizes": ["18.2x122 cm", "20.2x122 cm"], "dimensions": ["2x122"]}}
    out = sf.settle_product(product, [])
    assert out["metadata"]["available_sizes"] == ["18.2x122 cm"]
    assert "dimensions" not in out["metadata"]
    assert out["changes"]["sizes"]["dropped"] == ["20.2x122", "2x122"]


def test_a_packing_table_supports_a_size_and_a_spec_table_does_not():
    spec = {"table_data": {"headers": ["Variant", "PEI (ISO 10545-7)"], "rows": [["30x90", "IV"]]}}
    assert sf.table_formats([spec]) == []
    assert sf.table_formats([packing(GOOD_ROW), packing(["HEXA 23X27", "-", "16", "0,75"] + ["-"] * 8)]) == ["30x60", "23x27"]


def test_a_size_printed_right_after_the_name_stays_and_a_caption_beside_it_does_not():
    pages = ["| Passion Blue 25x25 cm | Hayedo Roble 23x120 cm |", "Tribeca\n23x120 cm"]
    assert sf.printed_sizes("Passion", pages) == {"25x25"}
    assert sf.printed_sizes("Tribeca", pages) == {"23x120"}
    product = {"name": "Passion", "description": "", "attributes": {}, "metadata": {"available_sizes": ["25x25 cm", "23x120 cm"]}}
    assert sf.settle_product(product, [], {"25x25"})["metadata"]["available_sizes"] == ["25x25 cm"]


def test_a_new_catalogue_keeps_a_size_only_its_extraction_found_when_the_page_prints_it():
    product = {"name": "Kale", "description": "x", "attributes": {}, "metadata": {"available_sizes": ["60x60 cm", "30x60 cm"], "available_formats": ["60x60"]}}
    assert sf.settle_product(product, [], {"30x60"})["metadata"]["available_formats"] == ["60x60", "30x60"]


def test_a_size_in_another_unit_is_left_alone():
    product = {"name": "Linear", "description": "x", "attributes": {}, "metadata": {"available_sizes": ["600x1200 mm"], "available_formats": ["60x60"]}}
    assert "600x1200 mm" in sf.settle_product(product, [])["metadata"]["available_sizes"]


def test_colours_are_the_same_on_both_sides_so_the_picker_offers_one():
    product = {"name": "Cento", "description": "x", "attributes": {"color": ["steel"]},
               "metadata": {"color": ["acero"], "colour_options": [{"name": "Acero", "hex": "#777777"}],
                            "series_reconcile": {"kept": {"colours": ["acero"]}}}}
    out = sf.settle_product(product, [])
    assert out["attributes"]["color"] == out["metadata"]["color"] == ["acero"]
    assert out["metadata"]["colour_options"] == [{"name": "Acero", "hex": "#777777"}]


def test_a_decor_only_series_loses_its_invented_colours_but_a_priced_swatch_survives():
    techno = {"name": "Techno", "description": "x", "attributes": {},
              "metadata": {"color": ["rombo", "dot"], "decorative_variant": ["dot", "rombo"],
                           "series_reconcile": {"kept": {"colours": [], "decors": ["dot", "rombo"]}}}}
    assert "color" not in sf.settle_product(techno, [])["metadata"]
    stella = {"name": "Stella", "description": "x", "attributes": {"color": ["oak"]},
              "metadata": {"color": ["oak"], "colour_options": [{"name": "Oak", "hex": "#aa8855"}], "series_reconcile": {"kept": {"colours": []}}}}
    assert sf.settle_product(stella, [])["metadata"]["color"] == ["oak"]


def test_the_rejected_colors_field_folds_into_color():
    product = {"name": "Blancos", "description": "x", "attributes": {}, "metadata": {"colors": ["Blanco Brillo", "Superwhite"]}}
    out = sf.settle_product(product, [])
    assert out["metadata"]["color"] == ["blanco brillo", "superwhite"] and "colors" not in out["metadata"]


def test_a_description_is_rebuilt_when_ours_or_wrong_and_prose_that_agrees_is_kept():
    ours = {"name": "Agata", "description": "Agata: floor tile, in 30x90 cm.", "attributes": {},
            "metadata": {"description_source": "reconcile", "available_formats": ["30x90", "59.6x59.6"]}}
    assert "59.6x59.6" in sf.settle_product(ours, [])["description"]
    agrees = {"name": "Abadia", "description": "A soft stone look in 50x50 cm.", "attributes": {}, "metadata": {"available_formats": ["50x50"]}}
    assert sf.settle_product(agrees, [])["description"] == agrees["description"]
    wrong = {"name": "Fuentede", "description": "Available in 40x120 cm.", "attributes": {}, "metadata": {"available_formats": ["30x60"]}}
    assert "40x120" not in sf.settle_product(wrong, [])["description"]


def test_settling_twice_changes_nothing():
    product = {"name": "Zen", "description": "", "attributes": {"color": ["steel"]},
               "metadata": {"available_formats": ["7.5x30"], "dimensions": [{"metric_cm": "5x25"}, {"metric_cm": "7.5x30"}],
                            "color": ["blanco"], "series_reconcile": {"kept": {"colours": ["blanco"]}}}}
    first = sf.settle_product(product, [])
    assert first["changes"]
    again = sf.settle_product({"name": "Zen", "description": first["description"], "attributes": first["attributes"], "metadata": first["metadata"]}, [])
    assert again["changes"] == {}


def test_the_audit_names_what_settling_leaves_impossible():
    bad = {"name": "Cento", "description": "", "attributes": {"color": ["steel"]},
           "metadata": {"color": ["acero"], "available_formats": ["23x27"], "available_sizes": ["23x27 cm", "40x120 cm"]}}
    assert set(sf.fact_problems(bad, [])) == {"colour_fields_disagree", "sizes_unverified", "description_missing"}
    settled = sf.settle_product(bad, [])
    assert sf.fact_problems({"name": "Cento", "description": settled["description"], "attributes": settled["attributes"], "metadata": settled["metadata"]}, []) == []


def test_two_printed_rows_run_together_are_caught_and_a_clean_row_is_not():
    shifted = ["33X33", "(K)", "14", "1,55", "26,80", "80X120", "42", "65,10", "1150", "(A)", "12", "1,33"]
    short = ['30x60 cm / 11,8"x23,6"', "WB (white body)", "3 mm joint"]
    wrong_maths = ["30X60", "-", "7", "1,26", "22,68", "103X103", "48", "90,00", "1100", "-", "-", "-"]
    table = packing(GOOD_ROW, shifted, short, wrong_maths)
    assert len(sf.table_cell_problems(table)) == 3
    assert sf.table_cell_problems(packing(GOOD_ROW)) == []


def test_a_list_stored_as_a_string_and_a_size_stored_as_a_dict_are_normalised():
    product = {"name": "Linear", "description": "x", "attributes": {"heating_mode": '["electric"]'},
               "metadata": {"width_mm": {"value": 600, "unit": "mm"}, "note": "[not json]", "_raw": '["keep"]'}}
    out = sf.settle_product(product, [])
    assert out["attributes"]["heating_mode"] == ["electric"]
    assert out["metadata"]["width_mm"] == 600 and out["metadata"]["note"] == "[not json]" and out["metadata"]["_raw"] == '["keep"]'


def test_a_warranty_is_read_into_months_and_an_unclear_one_is_not():
    assert sf.warranty_months("5 years") == 60 and sf.warranty_months("24 months") == 24
    assert sf.warranty_months("lifetime") is None and sf.warranty_months(None) is None
    out = sf.settle_product({"name": "L", "description": "x", "attributes": {}, "metadata": {}, "warranty": "5 years"}, [])
    assert out["columns"] == {"default_warranty_months": 60}
    assert sf.settle_product({"name": "L", "description": "x", "attributes": {}, "metadata": {}, "warranty": "5 years", "default_warranty_months": 24}, [])["columns"] == {}


def test_a_warranty_is_the_brands_only_when_every_product_stating_one_agrees():
    assert sf.brand_warranty([{"default_warranty_months": 60}, {"default_warranty_months": None}]) == 60
    assert sf.brand_warranty([{"default_warranty_months": 60}, {"default_warranty_months": 24}]) is None


INDEX_PAGE = "indice\n" + "\n".join(["30x90 cm / 11,8x35", "Agata", "Arco"] + [f"S{i}" for i in range(14)] + ["HEXA 23x27 cm", "Cento", "30x90 cm", "Cento"])


def test_index_pages_give_each_series_the_sizes_above_it_and_other_pages_give_nothing():
    names = ["Agata", "Arco", "Cento"] + [f"S{i}" for i in range(14)]
    out = sf.formats_on_index_pages([INDEX_PAGE, "Agata\n60x60 cm"], names)
    assert out["Agata"] == ["30x90"] and out["Cento"] == ["23x27", "30x90"]
    assert sf.formats_on_index_pages(["Agata\n60x60 cm\nArco"], names) == {}


def test_the_index_fills_a_gap_and_overrules_nothing():
    bare = {"name": "A", "description": "x", "attributes": {}, "metadata": {}}
    assert sf.settle_product(bare, [], (), ["30x90"])["metadata"]["available_formats"] == ["30x90"]
    held = {"name": "A", "description": "x", "attributes": {}, "metadata": {"available_formats": ["60x60"]}}
    assert sf.settle_product(held, [], (), ["30x90"])["metadata"]["available_formats"] == ["60x60"]
