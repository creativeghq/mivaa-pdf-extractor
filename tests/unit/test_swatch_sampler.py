import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "swatch_sampler_under_test", _ROOT / "app" / "services" / "products" / "swatch_sampler.py"
)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def W(text, x0, y0, x1, y1):
    return {"text": text, "rect": (x0, y0, x1, y1)}


def I(x0, y0, x1, y1):
    return {"rect": (x0, y0, x1, y1)}


def test_median_hex_and_spread():
    px = [(10, 20, 30)] * 5 + [(250, 250, 250)]
    assert S.median_hex(px) == "#0a141e"
    assert S.spread([(5, 5, 5)] * 4) == 0.0
    assert S.spread([(0, 0, 0), (100, 100, 100)]) == 50.0


def test_pattern_names():
    assert S.is_pattern_name("Peonies Warm")
    assert S.is_pattern_name("DECOR MIX")
    assert not S.is_pattern_name("Grey")


def test_containment_pairs_by_geometry_not_order():
    images = [I(0, 0, 50, 50), I(60, 0, 110, 50)]
    words = [W("DARK", 70, 20, 90, 30), W("WHITE", 10, 20, 40, 30)]
    pairs = S.pair_by_containment(words, images)
    assert [(p["label"], p["image"]["rect"][0]) for p in pairs] == [("WHITE", 0), ("DARK", 60)]


def test_containment_word_in_two_images_is_ambiguous():
    images = [I(0, 0, 50, 50), I(40, 0, 90, 50)]
    assert S.pair_by_containment([W("GREY", 42, 20, 48, 30)], images) == []


def test_caption_below_nearest_line_and_size_cut():
    images = [I(0, 100, 55, 114), I(60, 100, 115, 114)]
    words = [
        W("X1", 20, 118, 35, 126), W("X2", 80, 118, 95, 126),
        W("X9", 20, 160, 35, 168),
    ]
    pairs = S.pair_by_caption_below(words, images, max_gap=30)
    assert [p["label"] for p in pairs] == ["X1", "X2"]
    photo = [I(0, 0, 160, 55)]
    cap = [W("BIANCO", 5, 60, 40, 68), W("30x90", 45, 60, 70, 68)]
    assert S.pair_by_caption_below(cap, photo)[0]["label"] == "BIANCO"
    assert S.pair_by_caption_below([W("FAR", 5, 120, 40, 128)], photo) == []


def opts(*names, **extra):
    return [{"name": n, "catalogue_name": n, "hex": None, **extra} for n in names]


def sample(label, hexv, spread=3.0, **kw):
    return {"label": label, "hex": hexv, "spread": spread, "layout": "containment", **kw}


def test_merge_applies_and_never_overwrites():
    options = opts("Grey") + [{"name": "White", "hex": "#ffffff", "hex_source": "catalogue"}]
    new, report = S.merge_hex_into_colour_options(
        options, [sample("GREY", "#c0c0c0"), sample("WHITE", "#eeeeee")]
    )
    assert new[0]["hex"] == "#c0c0c0" and new[0]["hex_source"] == S.HEX_SOURCE
    assert new[1] == options[1]
    assert options[0]["hex"] is None
    assert [r["status"] for r in report] == ["applied", "skipped"]
    assert report[1]["reason"] == "has_hex"


def test_merge_reasons():
    options = opts("Peonies Warm", "Dark", "Blue", "Olive", "Grey")
    samples = [
        sample("DARK", "#111111", spread=50.0),
        sample("BLUE", "#0000ff"),
        sample("BLUE", "#2020ff"),
        sample("GREY", "#808080", reason="patterned"),
    ]
    new, report = S.merge_hex_into_colour_options(options, samples)
    assert [r.get("reason") for r in report] == [
        "pattern_name", "patterned", "ambiguous", "no_sample", "patterned"
    ]
    assert all(not o.get("hex") for o in new)


def test_merge_tie_within_tolerance_picks_first():
    new, _ = S.merge_hex_into_colour_options(
        opts("Blue"), [sample("BLUE", "#0000ff"), sample("BLUE", "#0202ff")]
    )
    assert new[0]["hex"] == "#0000ff"


def test_merge_matches_catalogue_name_and_phrase():
    options = [{"name": "Grigio", "hex": ""}]
    mixed = [sample("STRUTTURA GRIGIO", "#bbbbbb"), sample("GRIGIO", "#aaaaaa")]
    assert S.merge_hex_into_colour_options(options, mixed)[0][0]["hex"] == "#aaaaaa"
    phrase = S.merge_hex_into_colour_options(options, [sample("GRIGIO CHIARO", "#cccccc")])
    assert phrase[0][0]["hex"] == "#cccccc"


def test_sample_page_in_memory_pdf():
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    page = doc.new_page(width=600, height=400)
    for name, rgb, x in (("WHITE", (0.9, 0.9, 0.85), 20), ("DARK", (0.2, 0.25, 0.3), 100)):
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 50, 50), False)
        pix.set_rect(pix.irect, tuple(int(c * 255) for c in rgb))
        page.insert_image(fitz.Rect(x, 20, x + 50, 70), pixmap=pix)
        page.insert_text((x + 5, 27), name, fontsize=8)
    chip = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 55, 14), False)
    chip.set_rect(chip.irect, (180, 40, 40))
    page.insert_image(fitz.Rect(20, 200, 75, 214), pixmap=chip)
    page.insert_text((30, 226), "X4", fontsize=8)
    samples = {s["label"]: s for s in S.sample_page(None, 1, doc=doc)}
    assert set(samples) == {"WHITE", "DARK", "X4"}
    assert samples["WHITE"]["layout"] == "containment" and samples["X4"]["layout"] == "caption_below"
    assert samples["WHITE"]["hex"] == "#e5e5d8"
    assert samples["DARK"]["hex"] == "#333f4c"
    assert samples["X4"]["hex"] == "#b42828"
    assert "reason" not in samples["WHITE"]


def test_leaf_scale_keeps_a4_and_grows_for_spreads():
    assert S.leaf_scale(595, 842) == 1.0
    assert S.leaf_scale(842, 595) == pytest.approx(842 / 595)
    assert S.leaf_scale(1683.78, 666.14) == pytest.approx(841.89 / 595)


def test_spread_admits_keros_sized_tiles_and_not_full_page_photos():
    scale = S.leaf_scale(1683.78, 666.14)
    infos = [{"bbox": (903, 105, 1243, 218)}, {"bbox": (-1, -1, 843, 667)}]
    assert [c["rect"] for c in S._swatch_candidates(infos, scale)] == [(903, 105, 1243, 218)]
    assert S._swatch_candidates(infos) == []


def test_series_mode_rejects_another_series_and_two_colour_captions():
    options = [{"name": "Gris"}, {"name": "Beige"}]
    samples = [
        sample("Ferrara Gris", "#ecedea"),
        sample("Amalfi Beige | Gris", "#dbd7d2"),
        sample("Beige", "#d8cfc4"),
        sample("Amalfi Beige", "#d9d0c5"),
    ]
    new, report = S.merge_hex_into_colour_options(options, samples, series="Amalfi")
    assert not new[0].get("hex") and report[0]["reason"] == "no_sample"
    assert new[1]["hex"] == "#d8cfc4"


def test_series_mode_folds_accents():
    new, _ = S.merge_hex_into_colour_options([{"name": "Avena"}], [sample("ABADIA AVENA", "#c8b89a")], series="Abadía")
    assert new[0]["hex"] == "#c8b89a"
