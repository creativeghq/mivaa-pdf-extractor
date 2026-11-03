"""Drawing geometry over synthetic PDFs built with PyMuPDF in the test itself."""

from __future__ import annotations

import importlib.util
import math
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("numpy")
fitz = pytest.importorskip("pymupdf")


def _load_drawings():
    """Load the package by path: other unit tests replace `app` with stub modules."""
    name = "app.services.drawings"
    pkg = Path(__file__).resolve().parents[2] / "app" / "services" / "drawings"
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", "") == str(pkg / "__init__.py"):
        return existing
    for key in [k for k in sys.modules if k.startswith(name + ".")]:
        del sys.modules[key]
    spec = importlib.util.spec_from_file_location(name, pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_drawings = _load_drawings()
analyze_pdf, find_similar_in_pdf = _drawings.analyze_pdf, _drawings.find_similar_in_pdf
METRES_PER_PT = sys.modules["app.services.drawings.geometry"].METRES_PER_PT


def _font_file():
    for path in (
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        try:
            open(path, "rb").close()
            return path
        except OSError:
            continue
    return None


def _text(page, point, text, size=8):
    font = _font_file()
    if font:
        page.insert_text(point, text, fontsize=size, fontname="F0", fontfile=font)
    else:
        page.insert_text(point, text, fontsize=size)


def _filler(page, n=200, y=780):
    shape = page.new_shape()
    for k in range(n):
        x = 20 + (k % 100) * 0.5
        shape.draw_line((x, y), (x, y + 2))
    shape.finish(width=0.1)
    shape.commit()


def _quarter_arc(shape, cx, cy, r, pieces=1):
    if pieces == 1:
        p0, p3 = (cx + r, cy), (cx, cy + r)
        shape.draw_bezier(p0, (cx + r, cy + 0.5522847498 * r), (cx + 0.5522847498 * r, cy + r), p3)
        return
    for i in range(pieces):
        a0 = math.pi / 2 * i / pieces
        a1 = math.pi / 2 * (i + 1) / pieces
        h = 4 / 3 * math.tan((a1 - a0) / 4) * r
        p0 = (cx + r * math.cos(a0), cy + r * math.sin(a0))
        p3 = (cx + r * math.cos(a1), cy + r * math.sin(a1))
        c1 = (p0[0] - h * math.sin(a0), p0[1] + h * math.cos(a0))
        c2 = (p3[0] + h * math.sin(a1), p3[1] - h * math.cos(a1))
        shape.draw_bezier(p0, c1, c2, p3)


def _pdf(build, width=842, height=595, rotate=0) -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    build(page)
    if rotate:
        page.set_rotation(rotate)
    data = doc.tobytes()
    doc.close()
    return data


def _mpp(ratio):
    return METRES_PER_PT * ratio


class TestScale:
    def test_greek_scale_keyword(self):
        data = _pdf(lambda p: _text(p, (600, 560), "ΚΛΙΜΑΚΑ 1:50"))
        page = analyze_pdf(data)["pages"][0]
        assert page["scale"]["ratio"] == 50
        assert page["scale"]["units_per_page_width"] == pytest.approx(842 * METRES_PER_PT * 50, rel=1e-6)
        assert page["scale"]["plausible"] is True

    @pytest.mark.parametrize("text,ratio", [("Κλίμακα 1/100", 100), ("ΚΛ. 1:20", 20), ("SCALE 1 : 200", 200)])
    def test_variants(self, text, ratio):
        page = analyze_pdf(_pdf(lambda p: _text(p, (100, 100), text)))["pages"][0]
        assert page["scale"]["ratio"] == ratio

    def test_disagreement_is_noted(self):
        def build(p):
            _text(p, (100, 100), "ΚΛΙΜΑΚΑ 1:50")
            _text(p, (100, 300), "ΚΛΙΜΑΚΑ 1:50")
            _text(p, (500, 300), "SCALE 1:100")
        page = analyze_pdf(_pdf(build))["pages"][0]
        assert page["scale"]["ratio"] == 50
        assert any("disagree" in n and "1:100" in n for n in page["notes"])

    def test_absurd_ratio_rejected(self):
        page = analyze_pdf(_pdf(lambda p: _text(p, (100, 100), "SCALE 1:5000")))["pages"][0]
        assert page["scale"] is None

    def test_no_keyword_no_uncommon_ratio(self):
        page = analyze_pdf(_pdf(lambda p: _text(p, (100, 100), "Ref 1/37")))["pages"][0]
        assert page["scale"] is None


class TestPrintedRooms:
    def test_greek_label_with_name_and_enclosing_room(self):
        def build(p):
            p.draw_rect(fitz.Rect(100, 100, 400, 300))
            p.draw_rect(fitz.Rect(50, 50, 800, 550))
            _text(p, (200, 190), "ΚΟΥΖΙΝΑ")
            _text(p, (200, 205), "Ε=12,50 τ.μ.")
        page = analyze_pdf(_pdf(build))["pages"][0]
        rooms = [x for x in page["proposals"] if x["element_type"] == "room"]
        assert len(rooms) == 1
        room = rooms[0]
        assert room["label"] == "ΚΟΥΖΙΝΑ"
        assert room["value"] == 12.5
        assert room["unit"] == "m2" and room["kind"] == "area" and room["source"] == "printed"
        assert room["value_norm"] is None
        assert room["detail"]["polygon_area_norm"] == pytest.approx(300 * 200 / 842 ** 2, rel=1e-4)
        assert len(room["geometry"]["points"]) == 4
        assert room["geometry"]["points"][0] == {"x": round(100 / 842, 5), "y": round(100 / 595, 5)}
        assert "ΚΟΥΖΙΝΑ" in room["source_ref"]

    def test_label_on_furniture_takes_the_room_outline(self):
        mpp = _mpp(50)

        def build(p):
            _text(p, (650, 570), "ΚΛΙΜΑΚΑ 1:50")
            p.draw_rect(fitz.Rect(100, 100, 100 + 4 / mpp, 100 + 4 / mpp))
            p.draw_rect(fitz.Rect(150, 150, 150 + 1.6 / mpp, 150 + 2.0 / mpp))
            _text(p, (160, 180), "ΥΠΝΟΔΩΜΑΤΙΟ")
            _text(p, (160, 195), "E=16.00 m²")
        page = analyze_pdf(_pdf(build))["pages"][0]
        room = next(x for x in page["proposals"] if x["element_type"] == "room")
        assert room["detail"]["polygon_area_m2"] == pytest.approx(16.0, rel=1e-2)

    @pytest.mark.parametrize("text,value", [("E: 12.50m²", 12.5), ("Εμβ. 12,5 μ²", 12.5), ("12,50 m2", 12.5), ("8 τμ", 8.0)])
    def test_area_formats(self, text, value):
        page = analyze_pdf(_pdf(lambda p: _text(p, (300, 300), text)))["pages"][0]
        rooms = [x for x in page["proposals"] if x["element_type"] == "room"]
        assert [r["value"] for r in rooms] == [value]
        assert rooms[0]["label"] == "Room"

    def test_name_on_same_line(self):
        page = analyze_pdf(_pdf(lambda p: _text(p, (300, 300), "Υ/Δ 4,20 m2")))["pages"][0]
        rooms = [x for x in page["proposals"] if x["element_type"] == "room"]
        assert rooms[0]["label"] == "Υ/Δ" and rooms[0]["value"] == 4.2


class TestDoors:
    def _door_page(self, pieces=1, double=False):
        mpp = _mpp(50)
        r = 0.9 / mpp

        def build(p):
            _text(p, (650, 570), "ΚΛΙΜΑΚΑ 1:50")
            shape = p.new_shape()
            _quarter_arc(shape, 200, 200, r, pieces)
            if double:
                _quarter_arc(shape, 200, 200, r * 0.97, pieces)
            _quarter_arc(shape, 500, 300, r, pieces)
            shape.finish(width=0.3)
            shape.commit()
            circle = p.new_shape()
            circle.draw_circle((650, 150), r)
            circle.finish()
            circle.commit()
            _filler(p)
        return analyze_pdf(_pdf(build))["pages"][0]

    @pytest.mark.parametrize("pieces", [1, 2, 3])
    def test_quarter_arcs_are_doors(self, pieces):
        page = self._door_page(pieces)
        assert page["kind"] == "vector"
        doors = [x for x in page["proposals"] if x["element_type"] == "door"]
        assert len(doors) == 2
        assert all(d["label"] == "Door ~0.9 m" and d["value"] == 1 and d["unit"] == "pcs" for d in doors)
        assert all(d["confidence"] == 0.6 and len(d["geometry"]["points"]) == 4 for d in doors)

    def test_double_line_swing_dedupes(self):
        doors = [x for x in self._door_page(1, double=True)["proposals"] if x["element_type"] == "door"]
        assert len(doors) == 2

    def test_without_scale_lower_confidence(self):
        def build(p):
            shape = p.new_shape()
            _quarter_arc(shape, 200, 200, 15)
            shape.finish()
            shape.commit()
            _filler(p)
        page = analyze_pdf(_pdf(build))["pages"][0]
        doors = [x for x in page["proposals"] if x["element_type"] == "door"]
        assert len(doors) == 1 and doors[0]["label"] == "Door" and doors[0]["confidence"] == 0.35
        assert "walls need a scale" in page["notes"]


class TestWalls:
    def test_parallel_pairs_become_walls(self):
        mpp = _mpp(50)
        t = 0.2 / mpp
        length = 5.0 / mpp

        def build(p):
            _text(p, (650, 570), "ΚΛΙΜΑΚΑ 1:50")
            shape = p.new_shape()
            shape.draw_line((100, 100), (100 + length, 100))
            shape.draw_line((100, 100 + t), (100 + length, 100 + t))
            shape.draw_line((100, 200), (100, 200 + length * 0.8))
            shape.draw_line((100 + t, 200), (100 + t, 200 + length * 0.8))
            shape.finish(width=0.3)
            shape.commit()
            _filler(p)
        page = analyze_pdf(_pdf(build))["pages"][0]
        walls = [x for x in page["proposals"] if x["element_type"] == "wall"]
        assert len(walls) == 1
        wall = walls[0]
        assert wall["label"] == "Wall ~20 cm"
        assert wall["value"] is None and wall["unit"] == "m" and wall["kind"] == "length"
        assert wall["value_norm"] == pytest.approx(length * 1.8 / 842, rel=1e-3)
        assert len(wall["geometry"]["segments"]) == 2
        assert wall["value_norm"] * page["scale"]["units_per_page_width"] == pytest.approx(9.0, rel=1e-3)

    def test_outline_on_a_wall_face_is_not_a_second_wall(self):
        mpp = _mpp(50)
        t = 0.1 / mpp
        length = 5.0 / mpp

        def build(p):
            _text(p, (650, 570), "ΚΛΙΜΑΚΑ 1:50")
            shape = p.new_shape()
            shape.draw_line((100, 100), (100 + length, 100))
            shape.draw_line((100, 100 + t), (100 + length, 100 + t))
            shape.draw_line((100, 100), (100 + length * 0.6, 100))
            shape.draw_line((100, 100 + t), (100 + length * 0.6, 100 + t))
            shape.finish(width=0.3)
            shape.commit()
            _filler(p)
        page = analyze_pdf(_pdf(build))["pages"][0]
        walls = [x for x in page["proposals"] if x["element_type"] == "wall"]
        assert len(walls) == 1
        assert walls[0]["detail"]["length_m"] == pytest.approx(5.0, rel=1e-2)

    def test_one_face_opposite_two_segments_counts_its_whole_length(self):
        mpp = _mpp(50)
        t = 0.1 / mpp
        length = 5.0 / mpp

        def build(p):
            _text(p, (650, 570), "ΚΛΙΜΑΚΑ 1:50")
            shape = p.new_shape()
            shape.draw_line((100, 100), (100 + length, 100))
            shape.finish(width=0.3)
            shape.commit()
            for x0, x1 in ((100, 100 + length / 2 - 1), (100 + length / 2 + 1, 100 + length)):
                part = p.new_shape()
                part.draw_line((x0, 100 + t), (x1, 100 + t))
                part.finish(width=0.3)
                part.commit()
            _filler(p)
        page = analyze_pdf(_pdf(build))["pages"][0]
        walls = [x for x in page["proposals"] if x["element_type"] == "wall"]
        assert len(walls) == 1
        assert walls[0]["detail"]["length_m"] == pytest.approx(5.0, rel=1e-2)

    def test_reshaped_curve_changes_the_fingerprint(self):
        def build(bulge):
            def draw(p):
                shape = p.new_shape()
                shape.draw_bezier((100, 100), (150, 100 + bulge), (250, 100 + bulge), (300, 100))
                shape.finish(width=0.3)
                shape.commit()
                _filler(p)
            return draw
        a = analyze_pdf(_pdf(build(40)))["pages"][0]["fingerprint"]
        b = analyze_pdf(_pdf(build(120)))["pages"][0]["fingerprint"]
        assert a != b

    def test_too_complex_page_is_refused_not_read_as_empty(self, monkeypatch):
        monkeypatch.setattr(sys.modules["app.services.drawings.geometry"], "MAX_CONTENT_BYTES", 100)
        page = analyze_pdf(_pdf(lambda p: _filler(p)))["pages"][0]
        assert page["kind"] == "vector"
        assert page["proposals"] == [] and page["fingerprint"] == []
        assert any("measure it by hand" in n for n in page["notes"])

    def test_calibration_overrides(self):
        def build(p):
            shape = p.new_shape()
            shape.draw_line((100, 100), (400, 100))
            shape.draw_line((100, 106), (400, 106))
            shape.finish()
            shape.commit()
            _filler(p)
        page = analyze_pdf(_pdf(build), units_per_page_width=20.0)["pages"][0]
        walls = [x for x in page["proposals"] if x["element_type"] == "wall"]
        assert walls and walls[0]["label"] == "Wall ~15 cm"

    def test_large_page_is_fast(self):
        mpp = _mpp(50)

        def build(p):
            _text(p, (650, 570), "ΚΛΙΜΑΚΑ 1:50")
            shape = p.new_shape()
            for i in range(120):
                for j in range(60):
                    x, y = 20 + i * 6.5, 20 + j * 9
                    shape.draw_line((x, y), (x + 5, y))
                    shape.draw_line((x, y), (x, y + 0.3 / mpp))
            shape.finish(width=0.1)
            shape.commit()
        data = _pdf(build)
        start = time.time()
        analyze_pdf(data)
        assert time.time() - start < 20


class TestClassificationAndShape:
    def test_scan_and_empty(self):
        def build(p):
            pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 30), False)
            pix.clear_with(200)
            p.insert_image(p.rect, pixmap=pix)
        scan = analyze_pdf(_pdf(build))["pages"][0]
        assert scan["kind"] == "scan" and scan["image_coverage"] > 0.9 and scan["proposals"] == []
        empty = analyze_pdf(_pdf(lambda p: _text(p, (10, 10), "hello")))["pages"][0]
        assert empty["kind"] == "empty"

    def test_response_fields_and_fingerprint(self):
        def build(p):
            _filler(p)
            _text(p, (100, 100), "ΚΛΙΜΑΚΑ 1:50")
        data = _pdf(build)
        result = analyze_pdf(data)
        assert set(result) == {"page_count", "truncated", "pages"}
        page = result["pages"][0]
        assert set(page) == {
            "page", "width_pt", "height_pt", "aspect", "kind", "path_count", "text_chars",
            "image_coverage", "scale", "fingerprint", "notes", "proposals",
        }
        assert page["page"] == 1 and page["aspect"] == pytest.approx(842 / 595, rel=1e-5)
        assert len(page["fingerprint"]) == 256
        assert analyze_pdf(data)["pages"][0]["fingerprint"] == page["fingerprint"]
        other = analyze_pdf(_pdf(lambda p: _text(p, (100, 100), "ΚΛΙΜΑΚΑ 1:100")))["pages"][0]
        assert other["fingerprint"] != page["fingerprint"]

    def test_truncation(self):
        doc = fitz.open()
        for _ in range(3):
            doc.new_page()
        data = doc.tobytes()
        result = analyze_pdf(data, max_pages=2)
        assert result["page_count"] == 3 and result["truncated"] is True and len(result["pages"]) == 2

    def test_rotation_matches_viewer(self):
        def build(p):
            p.draw_rect(fitz.Rect(100, 100, 300, 200))
            _text(p, (150, 150), "Ε=10 τ.μ.")
        page = analyze_pdf(_pdf(build, rotate=90))["pages"][0]
        assert page["width_pt"] == 595 and page["height_pt"] == 842
        room = [x for x in page["proposals"] if x["element_type"] == "room"][0]
        xs = sorted(pt["x"] for pt in room["geometry"]["points"])
        ys = sorted(pt["y"] for pt in room["geometry"]["points"])
        assert xs[0] == pytest.approx((595 - 200) / 595, abs=1e-4)
        assert ys[0] == pytest.approx(100 / 842, abs=1e-4)

    def test_not_a_pdf(self):
        with pytest.raises(ValueError):
            analyze_pdf(b"not a pdf")


class TestFindSimilar:
    def _symbol(self, shape, x, y, s=10.0, rotate=False):
        if rotate:
            shape.draw_line((x, y), (x, y + 2 * s))
            shape.draw_line((x, y), (x + s, y))
            shape.draw_line((x + s, y), (x + s, y + 2 * s))
        else:
            shape.draw_line((x, y), (x + 2 * s, y))
            shape.draw_line((x, y), (x, y + s))
            shape.draw_line((x, y + s), (x + 2 * s, y + s))

    def _page(self):
        def build(p):
            shape = p.new_shape()
            for x, y in ((100, 100), (300, 100), (500, 400)):
                self._symbol(shape, x, y)
            self._symbol(shape, 600, 100, rotate=True)
            shape.draw_line((100, 300), (140, 300))
            shape.finish()
            shape.commit()
            _filler(p)
        return _pdf(build)

    def test_finds_repeats_including_rotated(self):
        data = self._page()
        box = [95 / 842, 95 / 595, 125 / 842, 115 / 595]
        result = find_similar_in_pdf(data, 1, box)
        assert result["reason"] is None and result["example_items"] == 3
        assert len(result["matches"]) == 4
        assert result["matches"][0]["score"] == 1.0
        assert all(len(m["geometry"]["points"]) == 4 for m in result["matches"])

    def test_empty_box_and_non_vector_page(self):
        data = self._page()
        assert find_similar_in_pdf(data, 1, [0.9, 0.05, 0.95, 0.1])["reason"] == "no_vector_content_in_box"
        blank = _pdf(lambda p: _text(p, (10, 10), "x"))
        assert find_similar_in_pdf(blank, 1, [0, 0, 1, 1])["reason"] == "not_a_vector_page"
        with pytest.raises(ValueError):
            find_similar_in_pdf(data, 5, [0, 0, 1, 1])
