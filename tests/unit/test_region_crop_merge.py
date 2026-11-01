"""A photo the layout model splits is cropped whole; separate photos stay separate."""

import fitz

from app.services.pdf.region_crop_extractor import merge_split_photo_regions


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
