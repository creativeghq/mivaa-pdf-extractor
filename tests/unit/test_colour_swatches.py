import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.products import colour_swatches as C  # noqa: E402


def s(label, hexv, page):
    return {"label": label, "hex": hexv, "spread": 3.0, "layout": "caption_below", "page": page}


def test_product_pages_collects_every_page_key():
    assert C.product_pages({"page_range": [65, 66], "catalogue_detail_page": "64", "_start_page": None}) == [64, 65, 66]


def test_every_colour_leaves_with_hex_or_a_reason():
    out = C.measure_product("Amalfi", [{"name": "Gris"}, {"name": "Beige"}, {"name": "Blanco", "hex": "#FFFFFF"}],
                            [s("Gris", "#dbdbd6", 66)], set(), "t")
    by = {o["name"]: o for o in out["colour_options"]}
    assert by["Gris"]["hex"] == "#dbdbd6" and "hex_status" not in by["Gris"]
    assert by["Beige"]["hex_status"] == "no_sample" and by["Beige"]["hex_checked_at"] == "t"
    assert by["Blanco"] == {"name": "Blanco", "hex": "#FFFFFF"}
    assert out["applied"] == 1 and out["unmeasured"] == ["Beige"]


def test_bare_caption_on_a_page_shared_with_another_series_is_not_trusted():
    out = C.measure_product("Amalfi", [{"name": "Gris"}], [s("Gris", "#dbdbd6", 64)], {64}, "t")
    assert out["colour_options"][0]["hex_status"] == "shared_page"
    named = C.measure_product("Amalfi", [{"name": "Gris"}], [s("Amalfi Gris", "#dbdbd6", 64)], {64}, "t")
    assert named["colour_options"][0]["hex"] == "#dbdbd6"
