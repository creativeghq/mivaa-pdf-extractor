"""The Stage 0 deadline leaves real headroom over what a healthy catalog takes."""

import ast
from pathlib import Path
from typing import Any, Dict

_SRC = Path(__file__).resolve().parents[2] / "app/utils/timeout_guard.py"


def _deadline():
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ProgressiveTimeoutStrategy")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "calculate_product_discovery_timeout")
    fn.decorator_list = []
    scope: Dict[str, Any] = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(_SRC), "exec"), scope)
    return scope["calculate_product_discovery_timeout"]


def test_a_71_page_catalog_gets_at_least_twice_the_time_a_healthy_run_took():
    # A healthy run of exactly this catalog took 5-6.5 minutes (330-390s) end to end.
    assert _deadline()(71, ["products", "certificates", "logos", "specifications"]) >= 2 * 390


def test_the_deadline_grows_with_pages_and_categories_and_is_capped():
    d = _deadline()
    assert d(100, ["products"]) > d(50, ["products"]) > d(10, ["products"])
    assert d(50, ["products", "logos"]) > d(50, ["products"])
    assert d(5000, ["products"] * 10) == 1800


def test_a_tiny_document_still_gets_a_real_floor():
    assert _deadline()(1, ["products"]) >= 180
