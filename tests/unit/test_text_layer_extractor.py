import importlib.util
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for _pkg in ("app", "app.services", "app.services.pdf"):
    sys.modules.setdefault(_pkg, types.ModuleType(_pkg))
_oc = _load("app.services.pdf.ocr_completeness", "app/services/pdf/ocr_completeness.py")
tl = _load("text_layer_under_test", "app/services/pdf/text_layer_extractor.py")


def words(n, text="alpha"):
    return [(10.0 + i, 10.0, 30.0 + i, 20.0, text) for i in range(n)]


def test_a_page_with_enough_clean_words_is_usable():
    v = tl.assess_text_layer(words(60))
    assert v.usable and v.reasons == [] and v.stats["words"] == 60


@pytest.mark.parametrize(
    "ws,coverage,reason",
    [
        (words(5), 0.0, "few_words"),
        (words(60, "caf�"), 0.0, "garbled_text"),
        (words(60, "(cid:12)"), 0.0, "garbled_text"),
        (words(60, ""), 0.0, "garbled_text"),
        (words(60), 0.95, "full_page_image"),
    ],
)
def test_an_unreliable_text_layer_is_not_used(ws, coverage, reason):
    v = tl.assess_text_layer(ws, coverage)
    assert not v.usable and reason in v.reasons


def numeric_words(n, y=100.0, x0=100.0):
    return [(x0 + i * 40, y, x0 + i * 40 + 30, y + 10, str(100 + i)) for i in range(n)]


def test_numbers_inside_ruled_tables_route_to_the_text_layer():
    ws = words(40) + numeric_words(30)
    v = tl.assess_text_layer(ws)
    rect = (90.0, 90.0, 90.0 + 30 * 40 + 40, 120.0)
    d = tl.route_page(v, ws, [rect])
    assert d.route == tl.ROUTE_TEXT_LAYER and d.stats["numeric_outside_tables"] == 0


def test_numbers_outside_any_table_keep_the_structural_ocr_with_a_reference():
    ws = words(40) + numeric_words(30)
    d = tl.route_page(tl.assess_text_layer(ws), ws, [])
    assert d.route == tl.ROUTE_OCR_WITH_REFERENCE and d.stats["unstructured_numeric_ratio"] == 1.0


def test_a_page_with_few_numbers_is_just_text():
    ws = words(40) + numeric_words(3)
    assert tl.route_page(tl.assess_text_layer(ws), ws, []).route == tl.ROUTE_TEXT_LAYER


def test_an_unusable_text_layer_goes_to_ocr_with_the_reason():
    d = tl.route_page(tl.assess_text_layer(words(3)), words(3), [])
    assert d.route == tl.ROUTE_OCR and d.reasons == ["few_words"]


def test_table_row_checks_flag_ragged_and_sparse_tables():
    assert tl.check_table_rows([["a", "b"], ["1", "2"], ["3", "4"]])["needs_review"] is False
    assert "ragged_rows" in tl.check_table_rows([["a", "b"], ["1"], ["3", "4", "5"], ["6"]])["reasons"]
    assert "sparse_cells" in tl.check_table_rows([["a", "", ""], ["", "", ""], ["", "", "1"]])["reasons"]
    assert "too_small" in tl.check_table_rows([["a", "b"]])["reasons"]


def test_cells_are_escaped_in_the_html():
    out = tl.rows_to_html([["<img src=x onerror=1>", None], ["a&b", "2"]])
    assert "<img" not in out and "&lt;img" in out and "a&amp;b" in out and "<td></td>" in out


CONTENT = {
    "blocks": [
        {"bbox": (50.0, 20.0, 300.0, 40.0), "text": "BASIC TECHNICAL PARAMETERS", "max_size": 18.0},
        {"bbox": (50.0, 60.0, 300.0, 70.0), "text": "A paragraph of body text.", "max_size": 9.0},
        {"bbox": (60.0, 120.0, 90.0, 130.0), "text": "inside", "max_size": 9.0},
        {"bbox": (50.0, 300.0, 300.0, 310.0), "text": "Footer note", "max_size": 9.0},
        {"bbox": (50.0, 80.0, 300.0, 90.0), "text": "More body text here.", "max_size": 9.0},
    ],
    "tables": [{"bbox": (50.0, 100.0, 300.0, 200.0), "rows": [["Type", "Q"], ["K10", "893"], ["K11", "1227"]]}],
    "images": [(320.0, 20.0, 400.0, 100.0)],
}


def test_elements_carry_the_ocr_shape_in_pixel_space():
    els = tl.elements_from_content(CONTENT, page_number=7, zoom=2.0, origin=(10.0, 5.0))
    by_type = {}
    for e in els:
        by_type.setdefault(e["region_type"], []).append(e)
    table = by_type["TABLE"][0]
    assert table["bbox"] == {"x": 80.0, "y": 190.0, "width": 500.0, "height": 200.0, "page": 7}
    assert table["metadata"]["source"] == "text-layer" and "<td>893</td>" in table["metadata"]["html"]
    assert table["text_content"].splitlines()[1] == "K10 | 893"
    assert [e["reading_order"] for e in els] == list(range(len(els)))
    assert all({"region_type", "bbox", "confidence", "reading_order", "text_content", "fragments", "metadata"} <= set(e) for e in els)


def test_text_inside_a_table_is_not_repeated_and_the_title_is_the_largest_short_block():
    els = tl.elements_from_content(CONTENT, 1, 1.0)
    texts = [e["text_content"] for e in els if e["region_type"] in ("TEXT", "TITLE")]
    assert "inside" not in texts
    assert [e["text_content"] for e in els if e["region_type"] == "TITLE"] == ["BASIC TECHNICAL PARAMETERS"]
    image = [e for e in els if e["region_type"] == "IMAGE"][0]
    assert image["text_content"] == ""


def test_a_table_takes_the_block_just_above_it_as_its_title():
    els = tl.elements_from_content(CONTENT, 1, 1.0)
    table = [e for e in els if e["region_type"] == "TABLE"][0]
    assert table["metadata"]["title"] == "More body text here."


def test_reference_numbers_are_taken_from_words_inside_a_rect():
    ws = [(0, 0, 10, 10, "893"), (0, 0, 10, 10, "1 227"), (500, 500, 510, 510, "9999"), (0, 0, 10, 10, "12")]
    assert tl.reference_numbers(ws, (0, 0, 100, 100)) == {"893", "227"}
    assert "9999" in tl.reference_numbers(ws)


def test_every_reader_accepts_both_layout_versions():
    assert tl.LAYOUT_PROCESSING_VERSIONS == frozenset({"paddleocr-vl", "text-layer"})


def test_a_real_pdf_with_a_ruled_table_routes_to_the_text_layer_and_keeps_every_number():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 60), "BASIC TECHNICAL PARAMETERS", fontsize=18)
    for i in range(30):
        page.insert_text((50, 100 + i * 8), "radiators are heated by hot water circulation", fontsize=6)
    x0, y0, cw, rh, rows, cols = 50, 400, 70, 22, 8, 5
    for r in range(rows + 1):
        page.draw_line((x0, y0 + r * rh), (x0 + cols * cw, y0 + r * rh))
    for c in range(cols + 1):
        page.draw_line((x0 + c * cw, y0), (x0 + c * cw, y0 + rows * rh))
    expected = set()
    for r in range(rows):
        for c in range(cols):
            text = ["Type", "H", "L", "Q", "W"][c] if r == 0 else str(100 + r * 10 + c * 111)
            if r:
                expected.add(text)
            page.insert_text((x0 + c * cw + 4, y0 + r * rh + 15), text, fontsize=9)
    content = tl.extract_page_content(page)
    verdict = tl.assess_text_layer(content["words"], content["image_coverage"])
    decision = tl.route_page(verdict, content["words"], [t["bbox"] for t in content["tables"]])
    assert decision.route == tl.ROUTE_TEXT_LAYER, decision
    els = tl.elements_from_content(content, 1, 250 / 72.0)
    table = [e for e in els if e["region_type"] == "TABLE"][0]
    assert expected <= {c for row in _oc.table_rows(table["metadata"]["html"]) for c in row}
    assert table["metadata"]["needs_review"] is False


def test_a_photo_heavy_page_keeps_its_ocr_reference_in_case_a_table_is_a_picture():
    ws = words(60)
    v = tl.assess_text_layer(ws, 0.3)
    d = tl.route_page(v, ws, [], largest_image=0.05)
    assert d.route == tl.ROUTE_OCR_WITH_REFERENCE and d.reasons == ["large_images_may_hold_text"]
    one = tl.route_page(tl.assess_text_layer(ws, 0.1), ws, [], largest_image=0.2)
    assert one.route == tl.ROUTE_OCR_WITH_REFERENCE


def block(x0, y0, x1=None):
    return {"bbox": (x0, y0, x1 if x1 is not None else x0 + 150, y0 + 12), "text": "t", "max_size": 9.0}


def test_columns_are_counted_from_narrow_blocks_that_start_apart():
    two = [block(40, 100 + 30 * i) for i in range(4)] + [block(320, 100 + 30 * i) for i in range(4)]
    assert tl.column_count(two, 600) == 2
    assert tl.column_count([block(40, 100 + 30 * i) for i in range(8)], 600) == 1
    assert tl.column_count([block(40, 100, 560)] * 8, 600) == 1


def test_a_two_column_page_is_read_by_ocr_so_products_are_not_interleaved():
    ws = words(60)
    two = [block(40, 100 + 30 * i) for i in range(4)] + [block(320, 100 + 30 * i) for i in range(4)]
    d = tl.route_page(tl.assess_text_layer(ws), ws, [], blocks=two, page_width=600)
    assert d.route == tl.ROUTE_OCR_WITH_REFERENCE and d.reasons == ["multi_column_layout"]


def test_cell_text_has_no_line_breaks_inside_a_row():
    assert tl.rows_to_text([["A1", "Ceramic tile" + chr(10) + "60x120", "12"]]) == "A1 | Ceramic tile 60x120 | 12"


def test_a_title_made_of_numbers_is_never_taken_as_a_table_title():
    blocks = [{"bbox": (50.0, 80.0, 300.0, 90.0), "text": "1780 1231 1017 675 481", "max_size": 9.0}]
    assert tl._table_title((50.0, 100.0, 300.0, 200.0), blocks) is None


def test_pdf_images_are_clipped_to_the_half_page_and_tiny_ones_dropped():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page(width=800, height=400)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10), False)
    pix.set_rect(pix.irect, (200, 100, 50))
    page.insert_image(fitz.Rect(300, 50, 500, 150), pixmap=pix)
    page.insert_image(fitz.Rect(10, 10, 20, 20), pixmap=pix)
    for i in range(40):
        page.insert_text((20, 200 + i * 4), "alpha beta gamma delta", fontsize=3)
    left = tl.extract_page_content(page, fitz.Rect(0, 0, 400, 400), include_tables=False)
    assert left["images"] == [(300.0, 50.0, 400.0, 150.0)] and left["tables"] == []
    right = tl.extract_page_content(page, fitz.Rect(400, 0, 800, 400), include_tables=False)
    assert right["images"] == [(400.0, 50.0, 500.0, 150.0)]
