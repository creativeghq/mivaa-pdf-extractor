"""A consensus page offset must not move a product whose start page was already right."""

import ast
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_SRC = Path(__file__).resolve().parents[2] / "app/services/discovery/product_discovery_service.py"


def _ranges():
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "anchored_start_ranges")
    scope: Dict[str, Any] = {"List": List, "Dict": Dict, "Tuple": Tuple, "Optional": Optional, "Callable": Callable}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(_SRC), "exec"), scope)
    return scope["anchored_start_ranges"]


def _end(_first: int) -> int:
    return 218


def test_a_start_already_right_stays_on_its_own_page_when_the_majority_shifts():
    # Izida 2026: two-page series came back one low, single-page series exact. Shifting all put Athens on Atlanta.
    entries = [(0, 68, 69), (1, 71, 71), (2, 72, 72), (3, 73, 73), (4, 81, 81), (5, 82, 83)]
    out = _ranges()(entries, 1, 231, _end)
    assert out[0] == [69, 70]
    assert out[1] == [71]
    assert out[2] == [72]
    assert out[3] == list(range(73, 81))
    assert out[4] == [81, 82]
    assert out[5] == list(range(83, 219))


def test_the_last_product_stops_at_its_section_end_rather_than_a_shifted_one():
    assert _ranges()([(0, 216, 217)], 1, 231, _end)[0] == [217, 218]


def test_no_heading_found_falls_back_to_the_consensus():
    out = _ranges()([(0, 10, None), (1, 12, None)], 1, 50, lambda first: 50)
    assert out[0] == [11, 12]
    assert out[1][0] == 13


def test_a_heading_far_from_the_shifted_start_is_a_name_collision_and_is_ignored():
    # Travertin matched on a Travertino page must not jump a whole section.
    out = _ranges()([(0, 160, 125), (1, 163, 163)], 1, 231, _end)
    assert out[0] == [161, 162]
    assert out[1][0] == 163


def test_products_sharing_a_start_page_both_keep_it():
    out = _ranges()([(0, 20, 21), (1, 20, 21), (2, 23, 24)], 1, 50, lambda first: 50)
    assert out[0] == out[1] == [21, 22, 23]


def test_a_start_past_the_document_end_is_left_out():
    assert 0 not in _ranges()([(0, 300, None)], 1, 231, _end)
