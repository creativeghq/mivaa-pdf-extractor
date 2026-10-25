"""A catalog over the plan's materials limit fails whole, before any product is created."""

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_STAGE_0 = _ROOT / "app/api/pdf_processing/stage_0_discovery.py"
_QUOTA = _ROOT / "app/services/products/material_quota.py"


def _stage_0_function():
    tree = ast.parse(_STAGE_0.read_text(encoding="utf-8"))
    return next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "process_stage_0_discovery"
    )


def test_the_quota_check_raises_before_the_creation_loop():
    fn = _stage_0_function()
    raise_lines = [
        n.lineno for n in ast.walk(fn)
        if isinstance(n, ast.Raise)
        and isinstance(n.exc, ast.Call)
        and getattr(n.exc.func, "id", None) == "MaterialQuotaExceeded"
    ]
    loop_lines = [
        n.lineno for n in ast.walk(fn)
        if isinstance(n, ast.For)
        and isinstance(n.iter, ast.Call)
        and ast.unparse(n.iter).startswith("enumerate(products_to_create")
    ]
    assert raise_lines, "stage 0 no longer refuses an over-limit catalog"
    assert loop_lines, "the creation loop moved; update this guard"
    assert max(raise_lines) < min(loop_lines)


def test_the_refusal_is_a_quota_error_the_rest_of_the_platform_recognises():
    tree = ast.parse(_QUOTA.read_text(encoding="utf-8"))
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    assert "MaterialQuotaExceeded" in classes
    assert [b.id for b in classes["MaterialQuotaExceeded"].bases] == ["RuntimeError"]
