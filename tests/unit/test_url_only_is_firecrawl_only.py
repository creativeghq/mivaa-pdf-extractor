"""
Guard: a `mode='url-only'` tracked query scrapes its pinned URL and nothing else.

WHY THIS EXISTS
---------------
"Custom Monitoring" promises the cheap thing: the user pastes one retailer URL and we
watch that page. Between 2026-05-01 and 2026-08-08 it did the expensive opposite.

`add_url_only()` stored `pinned_url` and **no backend code ever read it** — the only
reader in the whole repo was a React component using it as a display label. `refresh()`
had no `mode` branch, so the row fell through to the full discovery pass: Perplexity +
DataForSEO + the Greek/Idealo marketplaces + a batched Haiku identity classifier, keyed
on the product NAME. Then `due_for_refresh()` handed it to the hourly cron, so that pass
re-ran every 24h, forever, per pasted URL.

Nothing could see it. The row refreshed, prices appeared, the cron reported success —
they were simply prices for pages the user never asked about, while the page they DID
ask about went unfetched. No exception, no null, no failed constraint. The only symptom
was the bill, and with zero enrolled products there was no bill to notice (issue #234).

So the invariant is structural, not behavioural: the url-only path must be physically
incapable of reaching discovery. Source/AST based — imports no app module and touches no
DB, so it runs in CI in about a second.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[2]
_SERVICE = _ROOT / "app" / "services" / "integrations" / "tracked_queries_service.py"

#: Everything that costs money or an LLM call on the discovery path. None of it may be
#: reachable from `_refresh_url_only` — the user already chose the page, so there is
#: nothing to discover and no identity to adjudicate.
_DISCOVERY_CALLS = (
    "search_prices",
    "extract_query_facets",
    "_classify_and_filter",
    "classify_hits",
)


def _class_body(name: str) -> ast.ClassDef:
    tree = ast.parse(_SERVICE.read_text(encoding="utf-8"))
    node = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name),
        None,
    )
    assert node is not None, f"class {name} moved — update this guard rather than deleting it"
    return node


def _method(name: str) -> ast.AsyncFunctionDef:
    cls = _class_body("TrackedQueriesService")
    node = next(
        (
            n for n in cls.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
        ),
        None,
    )
    assert node is not None, f"{name}() moved — update this guard rather than deleting it"
    return node


def _called_attrs(node: ast.AST) -> set:
    """Every attribute name invoked as a call anywhere under `node`."""
    return {
        n.func.attr
        for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }


def test_refresh_routes_url_only_away_before_spending_anything():
    """The `mode` branch must come BEFORE the discovery call, not merely exist.

    A branch placed after `search_prices()` would read as correct and still bill for
    the full pass every time — which is the exact failure being guarded.
    """
    refresh = _method("refresh")

    handoffs = [
        n.lineno for n in ast.walk(refresh)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_refresh_url_only"
    ]
    assert handoffs, (
        "refresh() no longer hands `mode='url-only'` rows to _refresh_url_only(). "
        "Without that branch a pinned URL runs full Perplexity + DataForSEO + Haiku "
        "discovery on the product NAME and never fetches the URL the user pinned."
    )

    discovery = [
        n.lineno for n in ast.walk(refresh)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "search_prices"
    ]
    assert discovery, "refresh() no longer runs discovery at all — that is a bigger bug"

    assert min(handoffs) < min(discovery), (
        "the url-only branch sits AFTER search_prices(). It must short-circuit before "
        "any paid call, otherwise the row is billed for discovery it then discards."
    )

    # The branch must test `mode`, not something incidental that happens to correlate.
    assert "url-only" in ast.unparse(refresh), "the branch no longer tests mode == 'url-only'"


def test_url_only_path_cannot_reach_discovery():
    """Nothing under `_refresh_url_only` may call a discovery or classifier entry point."""
    fn = _method("_refresh_url_only")
    called = _called_attrs(fn)
    for forbidden in _DISCOVERY_CALLS:
        assert forbidden not in called, (
            f"_refresh_url_only calls {forbidden}(). The pinned-URL path is one "
            f"Firecrawl scrape; adding discovery back makes Custom Monitoring cost "
            f"more than the discovery flow it was meant to be cheaper than."
        )
    assert "_verify_hits_with_firecrawl" in called, (
        "_refresh_url_only no longer scrapes anything — the pinned URL would go unread, "
        "which is the 2026-05-01 bug verbatim."
    )


def test_the_pinned_url_is_actually_read():
    """`pinned_url` must be consumed here.

    It was written on insert and read by no backend code for three months. A column
    nothing reads is indistinguishable from a feature that does not exist.
    """
    fn = _method("_refresh_url_only")
    assert "pinned_url" in ast.unparse(fn), (
        "_refresh_url_only does not read pinned_url — the user's URL is being ignored"
    )


def test_url_only_still_gets_the_full_persistence_tail():
    """Cheap acquisition, identical bookkeeping.

    Skipping discovery must not also skip the sanity band, the alert dispatcher, the
    `current_*` cache or the cadence update — those are what make a monitored row
    behave like a monitored row.
    """
    assert "_persist_refresh" in _called_attrs(_method("_refresh_url_only")), (
        "_refresh_url_only bypasses _persist_refresh, so pinned URLs would get no "
        "anomaly banding, no alerts and no current_* cache"
    )
    assert "_persist_refresh" in _called_attrs(_method("refresh")), (
        "refresh() bypasses _persist_refresh — the two paths have forked again"
    )


def test_find_for_product_cannot_return_a_pinned_url_row():
    """`find_for_product` must pin `mode='discovery'`.

    A product owns one discovery row and N url-only siblings, all sharing its
    `product_id`. Without the filter this `LIMIT 1` returns whichever row Postgres
    feels like, so `/products/{id}/refresh`, `/track` and the exclusion routes can
    silently operate on a pinned URL instead of the product's discovery row.
    """
    fn = _method("find_for_product")
    filtered = any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "eq"
        and len(n.args) == 2
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == "mode"
        and isinstance(n.args[1], ast.Constant)
        and n.args[1].value == "discovery"
        for n in ast.walk(fn)
    )
    assert filtered, (
        "find_for_product() no longer filters mode='discovery' — it can return a "
        "Custom Monitoring row as though it were the product's tracked query"
    )
