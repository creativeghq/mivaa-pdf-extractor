"""A photo the layout model splits is cropped whole; separate photos stay separate."""

import importlib.util
import sys
from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("region_geometry_under_test", _ROOT / "app/services/pdf/region_geometry.py")
geo = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = geo
_spec.loader.exec_module(geo)


def _region(y0, y1, x0=100, x1=500, kind="IMAGE"):
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0, "region_type": kind}


def _identity(r):
    return fitz.Rect(r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"])


def test_two_regions_on_one_photo_become_one_crop_including_the_gap():
    photo = fitz.Rect(0, 0, 600, 900)
    out = geo.merge_split_photo_regions([_region(80, 480), _region(510, 700)], [photo], _identity)
    assert len(out) == 1
    assert (out[0]["y"], out[0]["y"] + out[0]["height"]) == (80, 700)
    assert out[0]["merged_from"] == 2


def test_two_photos_on_a_page_wide_background_stay_separate():
    background = fitz.Rect(0, 0, 600, 900)
    top, bottom = fitz.Rect(90, 70, 510, 490), fitz.Rect(90, 500, 510, 710)
    out = geo.merge_split_photo_regions([_region(80, 480), _region(510, 700)], [background, top, bottom], _identity)
    assert len(out) == 2


def test_regions_without_an_image_under_them_and_tables_are_untouched():
    table = _region(80, 480, kind="TABLE")
    out = geo.merge_split_photo_regions([table, _region(510, 700)], [], _identity)
    assert out == [table, _region(510, 700)]


def test_a_full_page_photo_the_layout_model_did_not_box_gets_its_own_crop():
    bound, zoom = fitz.Rect(0, 0, 600, 900), 2.0
    photo = {"rect": fitz.Rect(30, 90, 560, 800), "dpi": 72}
    out = geo.unboxed_photo_regions([], [photo], bound, zoom, _identity)
    assert len(out) == 1 and out[0]["source"] == "embedded_image_rect"
    assert (out[0]["x"], out[0]["y"], out[0]["width"]) == (60, 180, 1060)


def test_a_boxed_photo_or_a_small_swatch_gets_no_second_crop():
    bound = fitz.Rect(0, 0, 600, 900)
    photo = {"rect": fitz.Rect(30, 90, 560, 800), "dpi": 72}
    swatch = {"rect": fitz.Rect(10, 10, 60, 40), "dpi": 24}
    assert geo.unboxed_photo_regions([_region(100, 700, 40, 550)], [photo], bound, 1.0, _identity) == []
    assert geo.unboxed_photo_regions([], [swatch], bound, 1.0, _identity) == []


def test_heading_and_footer_printed_over_a_full_bleed_photo_are_cut_off():
    bound = fitz.Rect(0, 0, 600, 900)
    photo = {"rect": fitz.Rect(0, 0, 600, 900), "dpi": 72}
    heading, footer = fitz.Rect(40, 20, 400, 50), fitz.Rect(40, 860, 200, 880)
    out = geo.unboxed_photo_regions([], [photo], bound, 1.0, _identity, [heading, footer])
    assert (out[0]["y"], out[0]["y"] + out[0]["height"]) == (52, 858)


def test_source_quality_reads_the_pdf_raster_not_the_render():
    swatch = geo.assess_source_quality(fitz.Rect(0, 0, 166, 55), 24)
    assert swatch["verdict"] == "low_res" and swatch["sourcing"] == "pending" and swatch["native_px"] == [55, 18]
    photo = geo.assess_source_quality(fitz.Rect(0, 0, 595, 842), 300)
    assert photo["verdict"] == "ok" and "sourcing" not in photo
    assert geo.assess_source_quality(fitz.Rect(0, 0, 100, 100), None)["verdict"] == "vector"


def _spread_photo(x0, x1, half):
    full = fitz.Rect(x0, 0, x1, 500)
    return {"rect": full & half, "full": full, "dpi": 72}


def test_a_photo_across_the_fold_is_cropped_whole_by_the_half_holding_its_centre():
    left, right = fitz.Rect(0, 0, 567, 567), fitz.Rect(567, 0, 1134, 567)
    # A photo spanning x 0..680: its centre (340) lies on the left half.
    on_left = geo.gutter_photo_regions([_region(0, 500, 0, 567)], [_spread_photo(0, 680, left)], left, _identity)
    assert len(on_left) == 1 and on_left[0]["crosses_fold"] and on_left[0]["clip_pdf"] == (0, 0, 680, 500)
    # The right half holds only a 113pt sliver of it: dropped, not cropped twice.
    sliver = geo.gutter_photo_regions([_region(0, 500, 567, 680)], [_spread_photo(0, 680, right)], right, _identity)
    assert sliver == []


def test_a_photo_inside_one_half_and_a_page_wide_background_are_left_alone():
    left = fitz.Rect(0, 0, 567, 567)
    inside = {"rect": fitz.Rect(40, 40, 500, 400), "full": fitz.Rect(40, 40, 500, 400), "dpi": 72}
    assert geo.gutter_photo_regions([_region(40, 400, 40, 500)], [inside], left, _identity)[0].get("clip_pdf") is None
    background = _spread_photo(0, 1134, left)
    small = _region(100, 160, 100, 200)
    assert geo.gutter_photo_regions([small], [background], left, _identity) == [small]
