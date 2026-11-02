"""A small image is retired only on an answer that names it AND a real gallery photo."""

import importlib.util
import io
from pathlib import Path

import pytest

Image = pytest.importorskip("PIL.Image")

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("gallery_coverage_under_test", _ROOT / "app/services/images/gallery_coverage.py")
gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gc)


def test_only_a_covered_answer_naming_a_real_image_and_photo_counts():
    answer = {"images": [
        {"image": 1, "covered": True, "gallery_photo": 2},
        {"image": 2, "covered": False, "gallery_photo": None},
        {"image": 3, "covered": True, "gallery_photo": 9},
        {"image": 7, "covered": True, "gallery_photo": 1},
        {"image": 4, "covered": True, "gallery_photo": None},
    ]}
    assert gc.covered_indexes(answer, n_small=4, n_gallery=3) == {0: 1}


def test_split_names_the_covering_photo_and_keeps_the_rest():
    hit, rest = gc.split_covered(["s1", "s2", "s3"], {0: 1, 2: 0}, ["g1", "g2"])
    assert hit == {"s1": "g2", "s3": "g1"} and rest == ["s2"]


def test_the_sheet_holds_every_image_in_a_labelled_cell():
    imgs = [Image.new("RGB", (142, 142), (200, 180, 150)) for _ in range(5)]
    sheet = Image.open(io.BytesIO(gc.numbered_sheet(imgs, "S", cols=4)))
    assert sheet.size == (4 * gc.CELL, 2 * gc.CELL)
