"""Public tools: IP keying, fail-closed quota reservation, cache keys, failed-source scans."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
from fastapi import HTTPException  # noqa: E402
from starlette.requests import Request  # noqa: E402

APP = Path(__file__).resolve().parents[2] / "app"


class _Query:
    """Chainable stand-in for a supabase-py query; records every filter."""

    def __init__(self, table: "_Table"):
        self.table = table
        self.calls: list = []

    def __getattr__(self, name):
        def _chain(*args, **kwargs):
            self.calls.append((name, args))
            return self
        return _chain

    def execute(self):
        self.table.executed.append(self)
        if self.table.raise_on_execute:
            raise RuntimeError("db down")
        return self.table.response(self)


class _Table:
    def __init__(self):
        self.raise_on_execute = False
        self.executed: list = []
        self.response = lambda q: SimpleNamespace(data=[], count=0)

    def select(self, *a, **k):
        q = _Query(self)
        q.calls.append(("select", a))
        return q

    def insert(self, payload):
        q = _Query(self)
        q.calls.append(("insert", (payload,)))
        return q

    def update(self, payload):
        q = _Query(self)
        q.calls.append(("update", (payload,)))
        return q


def _stub(monkeypatch, name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _load(monkeypatch, name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def table(monkeypatch):
    t = _Table()
    client = SimpleNamespace(table=lambda name: t, rpc=lambda *a, **k: SimpleNamespace(execute=lambda: None))
    _stub(
        monkeypatch, "app.services.core.supabase_client",
        get_supabase_client=lambda: SimpleNamespace(client=client),
        read_rpc=lambda *a, **k: None,
    )
    return t


@pytest.fixture
def lookup(monkeypatch, table):
    return _load(
        monkeypatch, "app.services.integrations.public_lookup_service",
        APP / "services" / "integrations" / "public_lookup_service.py",
    )


@pytest.fixture
def routes(monkeypatch, lookup):
    class _Facets:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    _stub(monkeypatch, "app.services.integrations.perplexity_price_search_service",
          get_perplexity_price_search_service=lambda: None, PriceHit=object)
    _stub(monkeypatch, "app.services.integrations.mention_identity_service", SubjectFacets=_Facets)
    _stub(monkeypatch, "app.services.integrations.mention_search_service", MentionSearchService=object)
    _stub(monkeypatch, "app.services.integrations.platform_secret_resolver",
          resolve_secret=lambda key: SimpleNamespace(value=None))
    _stub(monkeypatch, "app.services.integrations.turnstile_verifier", verify_token=None)
    _stub(monkeypatch, "app.services.integrations.credit_router", debit_row=None)
    return _load(monkeypatch, "public_tools_routes_under_test", APP / "api" / "public_tools_routes.py")


def _request(headers: dict | None = None, client=("203.0.113.7", 5555)) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "POST", "path": "/", "headers": raw, "client": client})


def _open_quota(lookup, used=0):
    return lookup.QuotaStatus(
        allowed=used < lookup.ANONYMOUS_DAILY_QUOTA, used=used, limit=lookup.ANONYMOUS_DAILY_QUOTA,
        remaining=max(0, lookup.ANONYMOUS_DAILY_QUOTA - used), reset_at=datetime.now(timezone.utc),
    )


class _Recorder:
    def __init__(self):
        self.finalized: list = []
        self.cached: list = []
        self.refunds: list = []
        self.logged: list = []
        self.upstream_calls = 0


def _wire(monkeypatch, routes, lookup, *, user_id=None, quota_used=0, recheck_used=None,
          reserve_fails=False, debit=None, balance=10):
    rec = _Recorder()

    async def _verify(*a, **k):
        return SimpleNamespace(success=True, error_codes=[])

    def _check_quota(*, ip_address, user_id, created_up_to=None):
        used = recheck_used if (created_up_to and recheck_used is not None) else quota_used
        return _open_quota(lookup, used)

    def _reserve(**kw):
        if reserve_fails:
            raise lookup.QuotaUnavailable("insert refused")
        return lookup.Reservation(id="res-1", created_at="2026-11-19T10:00:00+00:00")

    monkeypatch.setattr(routes, "verify_token", _verify)
    monkeypatch.setattr(routes, "_resolve_user_id", lambda request: user_id)
    monkeypatch.setattr(routes, "check_quota", _check_quota)
    monkeypatch.setattr(routes, "reserve_scan", _reserve)
    monkeypatch.setattr(routes, "finalize_scan", lambda res, **kw: rec.finalized.append(kw["outcome"]) or True)
    monkeypatch.setattr(routes, "log_scan", lambda **kw: rec.logged.append(kw["outcome"]) or True)
    monkeypatch.setattr(routes, "read_cache", lambda **kw: None)
    monkeypatch.setattr(routes, "write_cache", lambda **kw: rec.cached.append(kw["scan_type"]))
    monkeypatch.setattr(routes, "_read_credit_balance", lambda uid: balance)
    monkeypatch.setattr(routes, "_debit_credits", lambda *a, **k: debit or (True, balance - 2, None))
    monkeypatch.setattr(routes, "_refund_credits", lambda uid, **kw: rec.refunds.append(kw["scan_type"]))
    return rec


def _price_service(monkeypatch, routes, rec, result):
    async def _search(**kw):
        rec.upstream_calls += 1
        return result

    monkeypatch.setattr(routes, "get_perplexity_price_search_service",
                        lambda: SimpleNamespace(search_prices=_search))


def _mention_service(monkeypatch, routes, rec, result):
    class _Svc:
        async def search(self, **kw):
            rec.upstream_calls += 1
            return result

    monkeypatch.setattr(routes, "MentionSearchService", _Svc)


def _price_result(*, success=True, hits=(), source_errors=None, error=None):
    errs = dict(source_errors or {})
    return SimpleNamespace(
        success=success, hits=list(hits), error=error, source_errors=errs, latency_ms=10,
        summary=None, credits_used=0,
        all_primary_sources_failed=lambda: all(s in errs for s in ("perplexity", "dataforseo")),
    )


def _price_body(routes):
    return routes.PublicPriceScanRequest(turnstile_token="t", product_name="Blum hinge")


def _mention_body(routes):
    return routes.PublicMentionScanRequest(turnstile_token="t", subject_label="Acme")


# ── (1) IP keying ────────────────────────────────────────────────────────────

def test_ip_is_the_socket_peer_not_a_caller_supplied_cf_header(routes):
    req = _request({"CF-Connecting-IP": "1.2.3.4"}, client=("198.51.100.9", 1))
    assert routes._extract_ip(req) == "198.51.100.9"


# ── (2)/(3) fail-closed quota + reservation ─────────────────────────────────

def test_check_quota_raises_when_the_count_cannot_be_read(lookup, table):
    table.raise_on_execute = True
    with pytest.raises(lookup.QuotaUnavailable):
        lookup.check_quota(ip_address="203.0.113.7", user_id=None)


def test_check_quota_counts_pending_and_only_its_own_scan_types(lookup, table):
    lookup.check_quota(ip_address="203.0.113.7", user_id=None)
    calls = table.executed[-1].calls
    in_filters = {args[0]: set(args[1]) for name, args in calls if name == "in_"}
    assert in_filters["outcome"] == {"success", "pending"}
    assert in_filters["scan_type"] == {"price", "mention", "product"}


def test_reservation_insert_failure_raises(lookup, table):
    table.raise_on_execute = True
    with pytest.raises(lookup.QuotaUnavailable):
        lookup.reserve_scan(scan_type="price", ip_address="203.0.113.7", user_id=None,
                            qhash="h", query_text="q")


def test_reservation_with_no_row_returned_raises(lookup, table):
    table.response = lambda q: SimpleNamespace(data=[], count=None)
    with pytest.raises(lookup.QuotaUnavailable):
        lookup.reserve_scan(scan_type="price", ip_address="203.0.113.7", user_id=None,
                            qhash="h", query_text="q")


def test_reservation_is_written_as_pending(lookup, table):
    table.response = lambda q: SimpleNamespace(data=[{"id": "r1", "created_at": "2026-11-19T10:00:00Z"}])
    res = lookup.reserve_scan(scan_type="price", ip_address="203.0.113.7", user_id=None,
                              qhash="h", query_text="q")
    assert res.id == "r1"
    payload = table.executed[-1].calls[0][1][0]
    assert payload["outcome"] == "pending" and payload["cache_hit"] is False


def test_unreadable_quota_is_503_before_any_upstream_call(monkeypatch, routes, lookup):
    rec = _wire(monkeypatch, routes, lookup)

    def _down(**kw):
        raise lookup.QuotaUnavailable("db down")

    monkeypatch.setattr(routes, "check_quota", _down)
    _price_service(monkeypatch, routes, rec, _price_result(hits=[object()]))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes.price_scan(_price_body(routes), _request()))
    assert exc.value.status_code == 503
    assert rec.upstream_calls == 0


def test_failed_reservation_is_503_before_any_upstream_call(monkeypatch, routes, lookup):
    rec = _wire(monkeypatch, routes, lookup, reserve_fails=True)
    _price_service(monkeypatch, routes, rec, _price_result(hits=[object()]))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes.price_scan(_price_body(routes), _request()))
    assert exc.value.status_code == 503
    assert rec.upstream_calls == 0


def test_losing_the_race_after_reserving_is_429_and_releases_the_slot(monkeypatch, routes, lookup):
    rec = _wire(monkeypatch, routes, lookup, quota_used=1, recheck_used=3)
    _price_service(monkeypatch, routes, rec, _price_result(hits=[object()]))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes.price_scan(_price_body(routes), _request()))
    assert exc.value.status_code == 429
    assert set(exc.value.detail) == {"message", "quota"}
    assert rec.finalized == ["rate_limited"]
    assert rec.upstream_calls == 0


# ── (8) debit failure is not "insufficient" ─────────────────────────────────

def test_debit_rpc_error_is_503_not_402(monkeypatch, routes, lookup):
    rec = _wire(monkeypatch, routes, lookup, user_id="u1", debit=(False, None, "Server disconnected"))
    _price_service(monkeypatch, routes, rec, _price_result(hits=[object()]))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes.price_scan(_price_body(routes), _request()))
    assert exc.value.status_code == 503
    assert rec.upstream_calls == 0


def test_short_balance_from_the_debit_is_402_with_quota(monkeypatch, routes, lookup):
    rec = _wire(monkeypatch, routes, lookup, user_id="u1",
                debit=(False, None, "Insufficient credits: balance 1, need 2"))
    _price_service(monkeypatch, routes, rec, _price_result(hits=[object()]))
    with pytest.raises(HTTPException) as exc:
        asyncio.run(routes.price_scan(_price_body(routes), _request()))
    assert exc.value.status_code == 402
    assert set(exc.value.detail) == {"message", "quota"}


def test_unreadable_balance_lets_the_debit_decide(monkeypatch, routes, lookup):
    rec = _wire(monkeypatch, routes, lookup, user_id="u1", balance=None, debit=(True, 8, None))
    _mention_service(monkeypatch, routes, rec, SimpleNamespace(
        hits=[], errors={}, by_source={"dataforseo_news": 0}, latency_ms=1, credits_used=0))
    out = asyncio.run(routes.mention_scan(_mention_body(routes), _request()))
    assert out.success is True
    assert rec.upstream_calls == 1


def test_zero_balance_still_gets_a_free_cache_hit(monkeypatch, routes, lookup):
    _wire(monkeypatch, routes, lookup, user_id="u1", balance=0)
    cached = {"success": True, "subject_label": "Acme", "country_code": None, "results": [],
              "total_results": 0, "top_outlets": [], "from_cache": False}
    monkeypatch.setattr(routes, "read_cache", lambda **kw: dict(cached))
    out = asyncio.run(routes.mention_scan(_mention_body(routes), _request()))
    assert out.from_cache is True
    assert out.quota.credits_per_scan == routes.SCAN_COST_BY_TYPE["mention"]


# ── (4) aliases are part of the mention cache key ───────────────────────────

def test_aliases_change_the_mention_cache_key(lookup):
    base = lookup.query_hash("mention", "Acme", "GR")
    with_alias = lookup.query_hash("mention", "Acme", "GR", aliases=["Acme Tiles", "ACME SA"])
    reordered = lookup.query_hash("mention", "Acme", "GR", aliases=["acme sa", " Acme  Tiles "])
    assert with_alias != base
    assert with_alias == reordered
    assert lookup.query_hash("mention", "Acme", "GR", aliases=[]) == base


# ── (5) mention: every source failed ────────────────────────────────────────

def _all_failed_mention():
    return SimpleNamespace(hits=[], errors={"dataforseo_news": "402", "perplexity_sonar": "timeout"},
                           by_source={}, latency_ms=5, credits_used=0)


def test_mention_all_sources_failed_signed_in_refunds_and_does_not_cache(monkeypatch, routes, lookup):
    rec = _wire(monkeypatch, routes, lookup, user_id="u1")
    _mention_service(monkeypatch, routes, rec, _all_failed_mention())
    out = asyncio.run(routes.mention_scan(_mention_body(routes), _request()))
    assert out.success is False and out.error
    assert rec.cached == []
    assert rec.refunds == ["mention"]


def test_mention_all_sources_failed_anonymous_releases_the_slot(monkeypatch, routes, lookup):
    rec = _wire(monkeypatch, routes, lookup)
    _mention_service(monkeypatch, routes, rec, _all_failed_mention())
    out = asyncio.run(routes.mention_scan(_mention_body(routes), _request()))
    assert out.success is False
    assert rec.finalized == ["failed"]
    assert rec.cached == []


def test_mention_partial_failure_serves_results_but_does_not_cache(monkeypatch, routes, lookup):
    rec = _wire(monkeypatch, routes, lookup)
    hit = SimpleNamespace(url="https://x.gr/a", title="t", excerpt="e", outlet_domain="x.gr",
                          outlet_name="X", published_at=None, source="news", language_code="el",
                          country_code="GR")
    _mention_service(monkeypatch, routes, rec, SimpleNamespace(
        hits=[hit], errors={"perplexity_sonar": "timeout"}, by_source={"dataforseo_news": 1},
        latency_ms=5, credits_used=0))
    out = asyncio.run(routes.mention_scan(_mention_body(routes), _request()))
    assert out.success is True and out.total_results == 1
    assert rec.cached == []
    assert rec.finalized == ["success"]


# ── (6) price: every source failed ──────────────────────────────────────────

def test_price_all_primary_sources_failed_refunds_and_does_not_cache(monkeypatch, routes, lookup):
    rec = _wire(monkeypatch, routes, lookup, user_id="u1")
    _price_service(monkeypatch, routes, rec, _price_result(
        success=True, source_errors={"perplexity": "not valid JSON", "dataforseo": "40501"}))
    out = asyncio.run(routes.price_scan(_price_body(routes), _request()))
    assert out.success is False and out.error
    assert rec.cached == []
    assert rec.refunds == ["price"]


def test_price_all_failed_anonymous_releases_the_slot(monkeypatch, routes, lookup):
    rec = _wire(monkeypatch, routes, lookup)
    _price_service(monkeypatch, routes, rec, _price_result(
        success=False, error="perplexity call failed",
        source_errors={"perplexity": "boom", "dataforseo": "boom"}))
    out = asyncio.run(routes.price_scan(_price_body(routes), _request()))
    assert out.success is False
    assert rec.finalized == ["failed"]


def test_price_with_any_source_error_is_not_cached(monkeypatch, routes, lookup):
    rec = _wire(monkeypatch, routes, lookup)
    hit = SimpleNamespace(retailer_name="R", product_url="https://r.gr/p", price=10.0,
                          original_price=None, currency="EUR", availability=None, verified=True,
                          source="dataforseo", product_title="t", match_kind="exact")
    _price_service(monkeypatch, routes, rec, _price_result(
        success=True, hits=[hit], source_errors={"dataforseo": "timeout"}))
    out = asyncio.run(routes.price_scan(_price_body(routes), _request()))
    assert out.success is True and len(out.results) == 1
    assert rec.cached == []
    assert rec.finalized == ["success"]


def test_clean_price_scan_is_cached_and_counted(monkeypatch, routes, lookup):
    rec = _wire(monkeypatch, routes, lookup)
    _price_service(monkeypatch, routes, rec, _price_result(success=True, hits=[]))
    out = asyncio.run(routes.price_scan(_price_body(routes), _request()))
    assert out.success is True
    assert rec.cached == ["price"]
    assert rec.finalized == ["success"]
