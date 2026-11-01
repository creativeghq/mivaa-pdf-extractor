"""A photo the layout model splits is cropped whole; separate photos stay separate."""

import pytest

fitz = pytest.importorskip("fitz")

from app.services.pdf.region_crop_extractor import merge_split_photo_regions  # noqa: E402


def _region(y0, y1, x0=100, x1=500, kind="IMAGE"):
    return {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0, "region_type": kind}


def _identity(r):
    return fitz.Rect(r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"])


def test_two_regions_on_one_photo_become_one_crop_including_the_gap():
    photo = fitz.Rect(0, 0, 600, 900)
    out = merge_split_photo_regions([_region(80, 480), _region(510, 700)], [photo], _identity)
    assert len(out) == 1
    assert (out[0]["y"], out[0]["y"] + out[0]["height"]) == (80, 700)
    assert out[0]["merged_from"] == 2


def test_two_photos_on_a_page_wide_background_stay_separate():
    background = fitz.Rect(0, 0, 600, 900)
    top, bottom = fitz.Rect(90, 70, 510, 490), fitz.Rect(90, 500, 510, 710)
    out = merge_split_photo_regions([_region(80, 480), _region(510, 700)], [background, top, bottom], _identity)
    assert len(out) == 2


def test_regions_without_an_image_under_them_and_tables_are_untouched():
    table = _region(80, 480, kind="TABLE")
    out = merge_split_photo_regions([table, _region(510, 700)], [], _identity)
    assert out == [table, _region(510, 700)]


def test_a_full_page_photo_the_layout_model_did_not_box_gets_its_own_crop():
    from app.services.pdf.region_crop_extractor import unboxed_photo_regions
    bound, zoom = fitz.Rect(0, 0, 600, 900), 2.0
    photo = {"rect": fitz.Rect(30, 90, 560, 800), "dpi": 72}
    out = unboxed_photo_regions([], [photo], bound, zoom, _identity)
    assert len(out) == 1 and out[0]["source"] == "embedded_image_rect"
    assert (out[0]["x"], out[0]["y"], out[0]["width"]) == (60, 180, 1060)


def test_a_boxed_photo_or_a_small_swatch_gets_no_second_crop():
    from app.services.pdf.region_crop_extractor import unboxed_photo_regions
    bound = fitz.Rect(0, 0, 600, 900)
    photo = {"rect": fitz.Rect(30, 90, 560, 800), "dpi": 72}
    swatch = {"rect": fitz.Rect(10, 10, 60, 40), "dpi": 24}
    assert unboxed_photo_regions([_region(100, 700, 40, 550)], [photo], bound, 1.0, _identity) == []
    assert unboxed_photo_regions([], [swatch], bound, 1.0, _identity) == []


def test_heading_and_footer_printed_over_a_full_bleed_photo_are_cut_off():
    from app.services.pdf.region_crop_extractor import unboxed_photo_regions
    bound = fitz.Rect(0, 0, 600, 900)
    photo = {"rect": fitz.Rect(0, 0, 600, 900), "dpi": 72}
    heading, footer = fitz.Rect(40, 20, 400, 50), fitz.Rect(40, 860, 200, 880)
    out = unboxed_photo_regions([], [photo], bound, 1.0, _identity, [heading, footer])
    assert (out[0]["y"], out[0]["y"] + out[0]["height"]) == (52, 858)
