"""Public-tools invariants checked from source, so they hold in CI where FastAPI is absent."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _blank_comments():
    spec = importlib.util.spec_from_file_location("comment_budget", _ROOT / "scripts" / "comment_budget.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.blank_comments


_BLANK = _blank_comments()
ROUTES = _BLANK((_ROOT / "app" / "api" / "public_tools_routes.py").read_text(encoding="utf-8"))
LOOKUP = _BLANK((_ROOT / "app" / "services" / "integrations" / "public_lookup_service.py").read_text(encoding="utf-8"))


def test_client_ip_never_comes_from_a_caller_header():
    assert "cf-connecting-ip" not in ROUTES.lower()
    body = ROUTES[ROUTES.index("def _extract_ip"):ROUTES.index("def _resolve_user_id")]
    assert "request.client" in body
    assert "headers" not in body


def test_quota_counts_only_its_own_scans_and_fails_closed():
    assert re.search(r"COUNTED_SCAN_TYPES\s*=\s*\(\s*\"price\",\s*\"mention\",\s*\"product\"\s*\)", LOOKUP)
    assert re.search(r"COUNTED_OUTCOMES\s*=\s*\(\s*\"success\",\s*\"pending\"\s*\)", LOOKUP)
    check = LOOKUP[LOOKUP.index("def check_quota"):LOOKUP.index("def reserve_scan")]
    assert "raise QuotaUnavailable" in check


def test_the_slot_is_reserved_before_the_paid_call():
    for route in ("async def price_scan", "async def mention_scan", "async def product_search"):
        start = ROUTES.index(route)
        body = ROUTES[start:ROUTES.index("\n@router", start) if "\n@router" in ROUTES[start:] else len(ROUTES)]
        admit = body.index("_admit(")
        upstream = min(i for i in (body.find("search_prices("), body.find(".search("), body.find("search_public_products")) if i >= 0)
        assert admit < upstream, route
