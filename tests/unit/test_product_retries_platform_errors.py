"""A Supabase platform hiccup retries the product; a product fault or a timeout does not."""

import ast
import asyncio
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "app/api/pdf_processing/parallel_product_processor.py"


def _load():
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    keep = [
        n for n in tree.body
        if (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in ("_TRANSIENT_BACKOFF_SECONDS", "_TRANSIENT_MARKERS"))
        or (isinstance(n, ast.FunctionDef) and n.name == "is_transient_platform_error")
    ]
    scope = {"asyncio": asyncio}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(_SRC), "exec"), scope)
    return scope


def test_schema_cache_and_db_timeouts_are_transient():
    check = _load()["is_transient_platform_error"]
    assert check(Exception("{'code': 'PGRST002', 'message': 'Could not query the database for the schema cache'}"))
    assert check(Exception("{'statusCode': 544, 'error': DatabaseTimeout}"))


def test_a_product_fault_and_the_per_product_timeout_are_not():
    check = _load()["is_transient_platform_error"]
    assert not check(ValueError("tile width missing"))
    assert not check(asyncio.TimeoutError("PGRST002"))


def test_the_retry_wraps_the_product_call_and_backs_off():
    scope = _load()
    assert len(scope["_TRANSIENT_BACKOFF_SECONDS"]) >= 2
    source = _SRC.read_text(encoding="utf-8")
    assert source.index("is_transient_platform_error(_exc)") > source.index("process_single_product(")
