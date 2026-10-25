"""A discovery range given as [first, last] is a span, never the two literal pages."""

import ast
from pathlib import Path
from typing import Any, Callable, Dict, List

_SRC = Path(__file__).resolve().parents[2] / "app/services/discovery/product_discovery_service.py"


def _expand():
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "expand_endpoint_page_ranges")
    scope: Dict[str, Any] = {"List": List, "Dict": Dict, "Callable": Callable}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(_SRC), "exec"), scope)
    return scope["expand_endpoint_page_ranges"]


def _end_of_document(_first: int) -> int:
    return 100


def test_a_first_last_pair_becomes_the_whole_span_up_to_the_next_product():
    expand = _expand()
    ranges = [[10, 17], [20, 25], [30, 33]]
    out = expand(ranges, 100, _end_of_document)
    assert out[0] == list(range(10, 20))
    assert out[1] == list(range(20, 30))
    assert out[2] == list(range(30, 101))


def test_the_span_is_never_shorter_than_the_range_the_model_gave():
    out = _expand()([[10, 40], [20, 25]], 100, _end_of_document)
    assert out[0][-1] == 40
    assert out[1][0] == 20


def test_a_full_list_a_consecutive_pair_and_an_empty_range_are_left_alone():
    out = _expand()([[10, 11, 12, 13], [20, 21], [], [30]], 100, _end_of_document)
    assert out == {}


def test_the_last_product_ends_where_its_section_ends():
    out = _expand()([[10, 14], [20, 24]], 100, lambda first: 29)
    assert out[1] == list(range(20, 30))


def test_the_span_stays_inside_the_document():
    out = _expand()([[90, 95]], 98, lambda first: 500)
    assert out[0][-1] == 98


def test_the_expansion_runs_before_start_pages_are_calculated():
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    method = next(
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "_calculate_page_ranges_from_start_pages" for c in ast.walk(n))
    )
    expand = [c.lineno for c in ast.walk(method) if isinstance(c, ast.Call) and getattr(c.func, "id", "") == "expand_endpoint_page_ranges"]
    calc = [c.lineno for c in ast.walk(method) if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "_calculate_page_ranges_from_start_pages"]
    assert expand and calc and max(expand) < min(calc)
