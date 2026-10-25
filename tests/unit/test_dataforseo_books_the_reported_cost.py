"""Guard: the cost booked for a DataForSEO call is the `cost` DataForSEO reports.

SERP calls were booked at a flat $0.0006 while a depth-100 live SERP is charged $0.0155, so
$33 of the $50 spent between 2026-10-10 and 10-25 never reached ai_usage_logs.
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
_BOOKED: list = []


def _by_path(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_client():
    saved = {k: sys.modules.get(k) for k in ("app", "app.services", _PKG, f"{_PKG}.dataforseo_envelope",
                                              f"{_PKG}.dataforseo_serp_targeting", f"{_PKG}.mention_cost_logger",
                                              f"{_PKG}.dataforseo_ai_parsing")}
    try:
        for name in ("app", "app.services", _PKG):
            pkg = types.ModuleType(name)
            pkg.__path__ = []
            sys.modules[name] = pkg
        sys.modules[_PKG].dataforseo_envelope = _by_path(f"{_PKG}.dataforseo_envelope", _INT / "dataforseo_envelope.py")
        sys.modules[_PKG].dataforseo_serp_targeting = _by_path(
            f"{_PKG}.dataforseo_serp_targeting", _INT / "dataforseo_serp_targeting.py")
        _by_path(f"{_PKG}.dataforseo_ai_parsing", _INT / "dataforseo_ai_parsing.py")
        logger = types.ModuleType(f"{_PKG}.mention_cost_logger")
        logger.CostAttribution = object
        logger.log_dataforseo_labs_call = logger.log_dataforseo_serp_call = lambda **k: _BOOKED.append(k)
        sys.modules[f"{_PKG}.mention_cost_logger"] = logger
        return _by_path("_dfs_client_cost_under_test", _INT / "dataforseo_unified_client.py")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


client_mod = _load_client()


def _body(task_status, cost, items):
    return {
        "status_code": 20000, "status_message": "Ok.", "tasks_count": 1,
        "tasks_error": 0 if task_status == 20000 else 1,
        "tasks": [{"status_code": task_status, "status_message": "x", "cost": cost, "result": [{"items": items}]}],
    }


def _run(monkeypatch, body, kind="serp"):
    class _Resp:
        status_code = 200
        text = "{}"

        def json(self):
            return body

    class _Http:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, *a, **k):
            return _Resp()

    monkeypatch.setattr(client_mod.httpx, "AsyncClient", _Http)
    c = client_mod.DataForSEOUnifiedClient(sandbox=False)
    c.b64 = "x"
    monkeypatch.setattr(c, "_charge_for_call", lambda *a, **k: 0)
    monkeypatch.setattr(c, "_refund_call", lambda *a, **k: None)
    _BOOKED.clear()
    asyncio.run(c._call("/x", [{}], log_kind=kind, operation="serp.google.organic:kw"))
    assert len(_BOOKED) == 1
    return _BOOKED[0]


def test_a_successful_call_books_the_reported_cost_not_a_constant(monkeypatch):
    booked = _run(monkeypatch, _body(20000, 0.0155, [{"x": 1}]))
    assert booked["cost_usd"] == pytest.approx(0.0155)
    assert booked["success"] is True


def test_a_partial_result_that_was_charged_is_booked_as_spend(monkeypatch):
    booked = _run(monkeypatch, _body(40106, 0.0155, [{"x": 1}]))
    assert booked["success"] is False
    assert booked["cost_usd"] == pytest.approx(0.0155), "DataForSEO charged for a partial SERP"


def test_labs_calls_book_the_reported_cost_too(monkeypatch):
    booked = _run(monkeypatch, _body(20000, 0.03, [{"x": 1}]), kind="labs")
    assert booked["cost_usd"] == pytest.approx(0.03)
