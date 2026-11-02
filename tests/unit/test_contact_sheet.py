"""A contact sheet is one row per product, so a missing or wrong image is visible at a glance."""

import importlib.util
import io
from pathlib import Path

import pytest

# CI installs pytest only; the contact sheet needs Pillow to render.
Image = pytest.importorskip("PIL.Image")

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("contact_sheet_under_test", _ROOT / "app/services/pdf/contact_sheet.py")
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


def _jpeg(size):
    out = io.BytesIO()
    Image.new("RGB", size, (120, 60, 30)).save(out, "JPEG")
    return out.getvalue()


def test_each_product_is_one_row_and_a_broken_image_still_takes_its_place():
    rows = [{"name": "Calma", "count": 8, "images": [_jpeg((1536, 1024)), None]},
            {"name": "Reality", "count": 0, "images": []}]
    sheet = Image.open(io.BytesIO(cs.render_sheet(rows)))
    assert sheet.height == cs.PAD + 2 * (cs.THUMB + cs.PAD)
    assert sheet.width == cs.LABEL_W + cs.THUMBS_PER_ROW * (cs.THUMB + cs.PAD) + cs.PAD
