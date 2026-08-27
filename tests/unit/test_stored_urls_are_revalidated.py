"""A URL that is STORED and RE-FETCHED ON A SCHEDULE needs the guard twice (mivaa#19 M6-2).

`test_ssrf_guard_coverage.py` already sweeps for unguarded server-side fetches. This
file holds the thing that sweep cannot express: for the price-monitoring URLs, passing
validation ONCE is not enough.

Everywhere else in the tree, a URL is fetched once in response to a request. Here the
URL is written to `tracked_queries.pinned_url` / `tracked_query_price_history.product_url`
and fetched again every refresh interval, indefinitely — so a single accepted write buys
a recurring internal-fetch primitive that outlives the session that created it, and DNS
can be re-pointed at an internal address at any point between the write and any of the
fetches that follow it.

Static, not runtime: CI installs pytest alone (`deploy.yml`) and these unit tests import
nothing from `app`, so each case parses source. A guard here proves the CALL is present,
not that the guard behaves — `assert_safe_url`'s own behaviour is covered by
`test_safe_fetch_bytes.py`.

Every case was watched to fail against the pre-fix source.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

SERVICE = APP / "services" / "integrations" / "tracked_queries_service.py"
ROUTES = APP / "api" / "price_monitoring_routes.py"

#: Every function that either stores one of these URLs or re-fetches one.
GUARDED_FUNCTIONS = [
    (SERVICE, "add_url_only"),        # the write
    (SERVICE, "_refresh_url_only"),   # the scheduled re-fetch of the pinned URL
    (SERVICE, "reverify"),            # the re-fetch of stored history rows
]


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _strip_comments(src: str) -> str:
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def _node(src: str, name: str):
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _source_of(src: str, name: str) -> str:
    return ast.get_source_segment(src, _node(src, name)) or ""


@pytest.mark.parametrize(("path", "func"), GUARDED_FUNCTIONS, ids=lambda v: getattr(v, "name", v))
def test_both_the_write_and_every_refetch_validate(path: Path, func: str):
    """Write-time validation alone does not close this: the fetches happen later, and
    the answer to "where does this host resolve?" can change in between."""
    body = _strip_comments(_source_of(_read(path), func))
    assert "assert_safe_url" in body, (
        f"{func} no longer validates its URL. For a stored, scheduled re-fetch that is "
        "not one missed check — it is a standing internal-fetch primitive (mivaa#19 M6-2)."
    )


def test_the_pinned_url_stored_is_the_validated_one():
    """Storing `url.strip()` while validating a separate local is a check that proves
    nothing about the row that gets written."""
    body = _strip_comments(_source_of(_read(SERVICE), "add_url_only"))
    assert '"pinned_url": safe_url' in body, (
        "the row is written with a URL other than the one that passed the guard"
    )


def test_a_refused_stored_url_does_not_disable_the_row():
    """`assert_safe_url` also raises when DNS resolution FAILS, so a transient outage is
    indistinguishable from a blocked host. Disabling on it would silently retire live
    monitoring — the failure has to be recorded and retried, not acted on."""
    body = _strip_comments(_source_of(_read(SERVICE), "_refresh_url_only"))
    at = body.find("assert_safe_url")
    assert at != -1
    window = body[at:at + 900]
    assert "last_error" in window, (
        "a refused pinned URL no longer records last_error, so the refusal is invisible"
    )
    assert '"disabled"' not in window and "is_active" not in window, (
        "the refusal path now disables the row. assert_safe_url raises on a DNS failure "
        "too, so that retires live monitoring on a transient outage."
    )


def test_refused_history_urls_are_reported_not_silently_dropped():
    """A silently shorter result set reads as "those retailers had nothing" rather than
    "we refused to fetch them" — the silent-zero shape."""
    body = _strip_comments(_source_of(_read(SERVICE), "reverify"))
    assert "refused" in body, "reverify no longer tracks the URLs it refused"
    assert '"refused_urls"' in body, (
        "reverify drops refused URLs from its result set without saying so"
    )


def test_the_route_refuses_before_it_charges():
    """Invariant 10, in the direction people forget: a URL we will refuse to fetch is
    not work, so the caller must not be debited for it and then refunded."""
    body = _strip_comments(_source_of(_read(ROUTES), "add_url_only"))
    guard_at = body.find("assert_safe_url")
    debit_at = body.find("metered_door(")
    assert guard_at != -1, "the route no longer validates the URL"
    assert debit_at != -1, "the route no longer meters"
    assert guard_at < debit_at, (
        "the URL check moved after the debit — the caller is now charged for a URL that "
        "is about to be refused"
    )
