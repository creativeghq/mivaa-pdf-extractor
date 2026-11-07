import importlib.util
import sys
import types
from pathlib import Path

_PRODUCTS = Path(__file__).resolve().parents[2] / "app" / "services" / "products"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_STUBS = ("app", "app.services", "app.services.products", "app.services.products.swatch_sampler")
_saved = {k: sys.modules.get(k) for k in _STUBS}
try:
    _pkg = types.ModuleType("app.services.products")
    _pkg.swatch_sampler = _load("swatch_sampler_for_vision_test", _PRODUCTS / "swatch_sampler.py")
    sys.modules.update({"app": types.ModuleType("app"), "app.services": types.ModuleType("app.services"),
                        "app.services.products": _pkg,
                        "app.services.products.swatch_sampler": _pkg.swatch_sampler})
    V = _load("swatch_vision_under_test", _PRODUCTS / "swatch_vision.py")
finally:
    for _k, _v in _saved.items():
        if _v is None:
            sys.modules.pop(_k, None)
        else:
            sys.modules[_k] = _v

A4 = (0.0, 0.0, 595.0, 842.0)


def test_izida_stacked_tiles_measure_only_their_visible_part():
    # Alba: framed WHITE chip, framed GREY chip drawn over its lower half, a room shot, an icon.
    rects = [(47, 213, 260, 426), (53, 218, 251, 416), (47, 320, 260, 476), (52, 324, 251, 466),
             (0, 592, 561, 807), (392, 127, 404, 144)]
    regions = V.candidate_regions(rects, A4)
    assert [r["rect"] for r in regions] == [(53, 218, 251, 416), (52, 324, 251, 466)]
    assert regions[0]["visible"] == (53, 218, 251, 324)
    assert regions[1]["visible"] == (52, 324, 251, 466)


def test_regions_on_the_other_half_of_a_spread_are_ignored():
    regions = V.candidate_regions([(100, 100, 200, 200), (700, 100, 800, 200)], (595.0, 0.0, 1190.0, 842.0))
    assert [r["rect"] for r in regions] == [(700, 100, 800, 200)]


def test_fully_covered_tile_is_not_offered():
    assert V.candidate_regions([(10, 10, 110, 110), (0, 0, 120, 120)], A4)[0]["rect"] == (0, 0, 120, 120)
    assert len(V.candidate_regions([(10, 10, 110, 110), (0, 0, 120, 120)], A4)) == 1


def test_picked_regions_keeps_only_asked_colours_and_real_numbers():
    answer = {"colours": [{"colour": "GREY", "regions": [2, 2, 9]}, {"colour": "Gold", "regions": [1]},
                          {"colour": "White", "regions": []}]}
    assert V.picked_regions(answer, ["Grey", "White"], 3) == {"Grey": [1]}


def test_apply_located_keeps_every_distinct_tile_and_refuses_a_pattern():
    options = [{"name": "Grey", "hex_status": "no_sample", "hex_checked_at": "x"}, {"name": "Decor", "hex_status": "no_sample"},
               {"name": "White", "hex": "#ffffff"}, {"name": "Gold", "hex_status": "no_sample"}]
    measured = {"Grey": [{"hex": "#808080", "spread": 10.0}, {"hex": "#818181", "spread": 9.0},
                         {"hex": "#707070", "spread": 12.0}],
                "Decor": [{"hex": "#333333", "spread": 90.0}]}
    out = V.apply_located(options, measured)
    assert out[0] == {"name": "Grey", "hex": "#808080", "hex_source": V.HEX_SOURCE, "hex_alternates": ["#707070"]}
    assert out[1]["hex_status"] == "patterned" and not out[1].get("hex")
    assert out[2] == options[2] and out[3] == options[3]
