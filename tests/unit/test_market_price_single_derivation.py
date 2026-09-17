"""The market price is derived once, in SQL, and every read of it records demand.

`resolve_market_price_from_hits` decides what a product is worth. Python re-deriving
a median beside it is the same defect as the finance one: the stored data stays
flawless, a wrong number is a valid number, and nothing raises.
"""
import ast
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
ROUTES = APP / "api" / "price_monitoring_routes.py"
LOOKUP = APP / "api" / "price_lookup_routes.py"
SERVICE = APP / "services" / "integrations" / "tracked_queries_service.py"
RESOLVER = APP / "services" / "integrations" / "market_price_resolver.py"


def _blank_comments(source: str) -> str:
    """Comments and docstrings blanked, so a rule can only be satisfied by code."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from comment_budget import blank_comments
        return blank_comments(source)
    finally:
        sys.path.pop(0)


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return _blank_comments(ast.get_source_segment(source, node) or "")
    raise AssertionError(f"{path.name} no longer defines {name}() — rename the guard with it")


def _load_resolver():
    spec = importlib.util.spec_from_file_location("market_price_resolver", RESOLVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_market_stats_delegates_to_the_sql_resolver():
    body = _function_source(ROUTES, "_compute_market_stats")
    assert "resolve_from_hits" in body, "_compute_market_stats must read the SQL resolver"


def test_market_stats_does_not_recompute_a_median_in_python():
    body = _function_source(ROUTES, "_compute_market_stats")
    for banned in ("sorted(", ".sort(", "len(values)", "median ="):
        assert banned not in body, f"{banned} re-derives the price in Python; SQL owns that"


def test_market_check_records_demand_on_both_paths():
    body = _function_source(ROUTES, "market_check")
    assert body.count("record_demand(") >= 2, (
        "the cached path is a request too — a read that records nothing looks like no demand"
    )


def test_partner_lookup_returns_the_band():
    body = _function_source(LOOKUP, "_claude_mode")
    assert "resolve_from_hits" in body, "search mode must return the min/max band and chosen price"


def test_refresh_queue_is_demand_ordered():
    body = _function_source(SERVICE, "due_for_refresh")
    assert "get_price_refresh_queue" in body, (
        "ordering by next_check_at alone ignores how often a price was actually asked for"
    )


def test_hits_to_payload_drops_unpriced_and_keeps_the_decision_inputs():
    mod = _load_resolver()
    out = mod.hits_to_payload([
        {"price": None, "retailer_name": "no price"},
        {"price": 42, "currency": "EUR", "availability": "in_stock", "verified": True,
         "retailer_name": "Shop", "product_url": "https://shop.example/p"},
    ])
    assert len(out) == 1
    row = out[0]
    assert row["price"] == "42", "money crosses as text so numeric precision survives"
    assert row["verified"] is True
    assert row["availability"] == "in_stock"
    for key in ("match_kind", "is_anomaly", "product_url", "retailer_name"):
        assert key in row, f"{key} decides the price; the resolver cannot see it if it is dropped"


def test_resolver_failure_is_unknown_not_no_data_and_never_a_zero():
    mod = _load_resolver()

    class Boom:
        def rpc(self, *a, **k):
            raise RuntimeError("db down")

    out = mod.resolve_from_hits(Boom(), [{"price": 10}])
    assert out["status"] == "collector_failed", (
        "a DB outage reading as 'no market price' lets the markup cap stop being enforced"
    )
    assert out["chosen_price"] is None, "a failed derivation must not read as a real price"


def test_nothing_to_derive_is_no_data():
    mod = _load_resolver()
    out = mod.resolve_from_hits(object(), [])
    assert out["status"] == "no_data"


def test_basis_is_passed_to_sql_and_an_unknown_one_falls_back():
    mod = _load_resolver()
    seen = {}

    class Spy:
        def rpc(self, name, args):
            seen["name"] = name
            seen["basis"] = args.get("p_basis")
            return self

        def execute(self):
            class R:
                data = dict(mod.EMPTY)
            return R()

    mod.resolve_from_hits(Spy(), [{"price": 10}], "highest")
    assert seen["name"] == "resolve_market_price_from_hits"
    assert seen["basis"] == "highest", "the basis must reach SQL; picking it in Python is a second derivation"

    mod.resolve_from_hits(Spy(), [{"price": 10}], "lowset")
    assert seen["basis"] == "verified_in_stock", "a typo must not silently become a different figure"


def test_money_is_not_rounded_through_a_binary_float():
    mod = _load_resolver()
    from decimal import Decimal
    out = mod.hits_to_payload([{"price": Decimal("9007199254740993.01")}])
    assert out[0]["price"] == "9007199254740993.01", (
        "float() loses the cents before Postgres ever casts to numeric"
    )


def test_every_offered_basis_is_one_sql_accepts():
    mod = _load_resolver()
    assert set(mod.BASES) == {"verified_in_stock", "median", "lowest", "highest"}
