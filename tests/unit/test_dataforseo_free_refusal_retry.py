"""Guard: DataForSEO's free 402/429 refusals are retried, and the retry is not billed again.

A funded account still had ~1 in 5 calls refused with 402 (2026-10-10); without the
retry 90 of 131 tracked keywords read "unknown" after a top-up.
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("httpx")

_INT = Path(__file__).resolve().parents[2] / "app" / "services" / "integrations"
_PKG = "app.services.integrations"


def _by_path(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_client():
    """Load the client by path: the package __init__ pulls in supabase, absent in CI."""
    saved = {k: sys.modules.get(k) for k in ("app", "app.services", _PKG, f"{_PKG}.dataforseo_envelope",
                                              f"{_PKG}.dataforseo_serp_targeting", f"{_PKG}.mention_cost_logger")}
    try:
        for name in ("app", "app.services", _PKG):
            pkg = types.ModuleType(name)
            pkg.__path__ = []
            sys.modules[name] = pkg
        sys.modules[_PKG].dataforseo_envelope = _by_path(f"{_PKG}.dataforseo_envelope", _INT / "dataforseo_envelope.py")
        sys.modules[_PKG].dataforseo_serp_targeting = _by_path(
            f"{_PKG}.dataforseo_serp_targeting", _INT / "dataforseo_serp_targeting.py")
        logger = types.ModuleType(f"{_PKG}.mention_cost_logger")
        logger.CostAttribution = object
        logger.log_dataforseo_labs_call = logger.log_dataforseo_serp_call = lambda *a, **k: None
        sys.modules[f"{_PKG}.mention_cost_logger"] = logger
        return _by_path("_dfs_client_under_test", _INT / "dataforseo_unified_client.py")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


client_mod = _load_client()

OK_BODY = {
    "status_code": 20000, "status_message": "Ok.", "tasks_count": 1, "tasks_error": 0,
    "tasks": [{"status_code": 20000, "status_message": "Ok.", "cost": 0.01, "result": [{"items": [{"x": 1}]}]}],
}


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body


def _client(monkeypatch, statuses):
    calls = {"http": 0, "charged": 0, "refunded": 0}
    queue = list(statuses)

    class _Http:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            calls["http"] += 1
            status = queue.pop(0)
            return _Resp(status, OK_BODY if status == 200 else {"status_code": 40200, "status_message": "Payment Required."})

        get = post

    async def _no_sleep(*a, **k):
        return None

    monkeypatch.setattr(client_mod.httpx, "AsyncClient", _Http)
    monkeypatch.setattr(client_mod.asyncio, "sleep", _no_sleep)
    c = client_mod.DataForSEOUnifiedClient(sandbox=False)
    c.b64 = "x"

    def _charge(*a, **k):
        calls["charged"] += 1
        return 1

    def _refund(*a, **k):
        calls["refunded"] += 1

    monkeypatch.setattr(c, "_charge_for_call", _charge)
    monkeypatch.setattr(c, "_refund_call", _refund)
    monkeypatch.setattr(c, "_log_cost", lambda *a, **k: None)
    return c, calls


def test_a_transient_402_is_retried_to_success_and_charged_once(monkeypatch):
    c, calls = _client(monkeypatch, [402, 402, 200])
    r = asyncio.run(c._call("/x", [{}]))
    assert r.ok and r.items == [{"x": 1}]
    assert calls == {"http": 3, "charged": 1, "refunded": 0}


def test_a_402_that_persists_fails_and_refunds(monkeypatch):
    c, calls = _client(monkeypatch, [402, 402, 402, 402])
    r = asyncio.run(c._call("/x", [{}]))
    assert not r.ok and r.status_code == 402
    assert calls["http"] == 4 and calls["charged"] == 1 and calls["refunded"] == 1
