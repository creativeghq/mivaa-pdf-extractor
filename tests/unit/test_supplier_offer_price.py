"""A distributor offer carries the catalogue's printed price, but only with a stated currency."""

import ast
from pathlib import Path
from typing import Any, Optional

_SRC = Path(__file__).resolve().parents[2] / "app/api/pdf_processing/stage_0_discovery.py"


def _parse():
    tree = ast.parse(_SRC.read_text(encoding="utf-8"))
    keep = [n for n in tree.body if (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_CURRENCY_SYMBOLS")
            or (isinstance(n, ast.FunctionDef) and n.name == "offer_price_from_metadata")]
    scope = {"Any": Any, "Optional": Optional}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(_SRC), "exec"), scope)
    return scope["offer_price_from_metadata"]


def test_a_plain_number_and_a_currency_code_are_taken_as_printed():
    f = _parse()
    assert f({"supplier_price": 12.5, "price_currency": "eur"}) == (12.5, "EUR")
    assert f({"supplier_price": {"value": "18"}, "price_currency": {"value": "USD"}}) == (18.0, "USD")


def test_european_and_us_number_formats_both_read_correctly():
    f = _parse()
    assert f({"supplier_price": "1.234,56", "price_currency": "EUR"})[0] == 1234.56
    assert f({"supplier_price": "1,234.56", "price_currency": "EUR"})[0] == 1234.56
    assert f({"supplier_price": "12,50 €", "price_currency": "EUR"})[0] == 12.5
    assert f({"supplier_price": "1,234", "price_currency": "EUR"})[0] == 1234.0


def test_a_symbol_is_a_currency_and_a_missing_currency_records_no_price():
    f = _parse()
    assert f({"supplier_price": "9.9", "price_currency": "€"}) == (9.9, "EUR")
    assert f({"supplier_price": 9.9}) == (None, None)
    assert f({"price_currency": "EUR"}) == (None, None)


def test_garbage_zero_and_absurd_values_record_nothing():
    f = _parse()
    for bad in ("on request", "", "0", "-5", 5_000_000, None):
        assert f({"supplier_price": bad, "price_currency": "EUR"}) == (None, None), bad
    assert f({"supplier_price": 3, "price_currency": "EURO"}) == (None, None)
    assert f(None) == (None, None)


def test_the_offer_call_passes_the_parsed_price_and_currency():
    src = _SRC.read_text(encoding="utf-8")
    call = src[src.index("record_catalog_supplier_offer"):]
    assert "'p_cost': cost" in call and "'p_currency': currency" in call
