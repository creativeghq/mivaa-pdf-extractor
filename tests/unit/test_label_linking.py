"""An image on a shared page goes to the series its printed label names, and to nothing on a guess."""

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("label_linking_under_test", _ROOT / "app/services/images/label_linking.py")
ll = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = ll
_spec.loader.exec_module(ll)

SERIES = {" ".join(t): t for t in map(ll.tokens_of, ["Ardenza", "Calma", "Wood", "Wood Espiga", "Habitat"])}


def _line(x0, y0, x1, y1, text):
    return ll.Line(x0, y0, x1, y1, ll.tokens_of(text))


def test_each_swatch_takes_the_label_printed_under_it():
    lines = [_line(0.10, 0.42, 0.35, 0.45, "ARDENZA 60x120"), _line(0.55, 0.42, 0.80, 0.45, "Calma Blanco 30x60")]
    assert ll.label_for([0.10, 0.10, 0.25, 0.30], lines, SERIES) == "ardenza"
    assert ll.label_for([0.55, 0.10, 0.25, 0.30], lines, SERIES) == "calma"


def test_a_longer_series_name_wins_and_a_line_naming_two_series_decides_nothing():
    assert ll.label_for([0.1, 0.1, 0.3, 0.3], [_line(0.1, 0.42, 0.4, 0.45, "WOOD ESPIGA Oak")], SERIES) == "wood espiga"
    two = [_line(0.1, 0.42, 0.4, 0.45, "Ardenza / Calma"), _line(0.1, 0.9, 0.4, 0.93, "Habitat")]
    assert ll.label_for([0.1, 0.1, 0.3, 0.3], two, SERIES) is None


def test_with_no_label_near_it_the_half_page_heading_decides_only_when_it_names_one_series():
    far = [_line(0.05, 0.02, 0.5, 0.06, "HABITAT"), _line(0.6, 0.95, 0.9, 0.98, "Habitat Oak 20x120")]
    assert ll.label_for([0.3, 0.3, 0.3, 0.3], far, SERIES) == "habitat"
    assert ll.label_for([0.3, 0.3, 0.3, 0.3], far + [_line(0.6, 0.97, 0.9, 0.99, "Ardenza")], SERIES) is None


def test_a_line_repeated_on_every_page_is_furniture_and_never_a_label():
    hidden = [(10, 10, 90, 20, "Barcelona", 0, 0, 0), (95, 10, 140, 20, "Arch", 0, 0, 1)]
    pages = [hidden + [(10, 50, 60, 60, f"page{i}", 1, 0, 0)] for i in range(12)]
    assert ll.furniture_lines(pages) == {("barcelona", "arch")}
    lines = ll.half_lines(pages[0], 200, 100, "left", ll.furniture_lines(pages))
    assert [l.tokens for l in lines] == [("page0",)]


def test_a_single_link_moves_only_off_a_page_that_prints_other_series_and_never_its_own():
    others = [_line(0.1, 0.42, 0.35, 0.45, "ARDENZA 25x25"), _line(0.55, 0.42, 0.8, 0.45, "HABITAT 25x25")]
    assert ll.names_only_others(others, SERIES, "calma")
    assert not ll.names_only_others(others + [_line(0.1, 0.02, 0.5, 0.06, "CALMA")], SERIES, "calma")
    assert not ll.names_only_others([_line(0.1, 0.9, 0.4, 0.93, "25x25 cm")], SERIES, "calma")
    assert not ll.names_only_others(others, SERIES, None)


def test_an_image_with_no_position_goes_to_the_one_series_its_page_prints():
    titan = [_line(0.1, 0.1, 0.4, 0.13, "TITAN GREIGE 30X90 CM"), _line(0.5, 0.1, 0.9, 0.13, "DECORADO TITAN VERDE")]
    series = {" ".join(t): t for t in map(ll.tokens_of, ["Tiber", "Titan"])}
    assert ll.names_only_others(titan, series, "tiber") and ll.sole_series(titan, series) == "titan"
    assert ll.sole_series(titan + [_line(0.1, 0.9, 0.4, 0.93, "TIBER 30X90")], series) is None
