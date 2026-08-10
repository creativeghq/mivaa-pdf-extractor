"""Guard: every door onto a paid job-research refresh debits before it spends.

The bug this exists to stop: `svc.refresh()` runs a paid fan-out — DataForSEO SERP, Perplexity
Sonar, Firecrawl career-page scrapes, Haiku classification. It had THREE callers:

    job_tracking_routes.py   partner API key   debits 5 credits up front   ✅
    job_research_routes.py   /cron-refresh     charge_cron per subject     ✅
    job_research_routes.py   /track/{id}/refresh  (the button in the app)  ❌ nothing

Two doors onto one paid operation, one of which checks. Nothing could see it: the unmetered
door returns 200 with real listings, the cost lands in `ai_usage_logs` exactly as it should,
and the only trace of the problem is a number that stays at zero in a different table.

`ops.silent_zero` cannot catch this one either. Its `ai_spend_never_debited` probe reads
`ai_usage_logs.credits_debited`, and job-research charges through `credit_transactions`
instead — so a module that meters perfectly and one that does not meter at all look identical
from there. (That probe now defers to `cron_metering:*` for cron-metered modules; this test
covers the interactive doors it still cannot see.)

SCOPE. This reads source text, so it pins the SHAPE of each route, not its runtime behaviour.
It cannot tell you the debit succeeded — only that the code cannot spend before trying.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROUTES = ROOT / "app" / "api" / "job_research_routes.py"
TRACKING_ROUTES = ROOT / "app" / "api" / "job_tracking_routes.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def _route_body(src: str, decorator: str) -> str:
    """Source of the handler under `decorator`, up to the next @router decorator."""
    start = src.index(decorator)
    nxt = src.find("@router.", start + len(decorator))
    return src[start : nxt if nxt != -1 else len(src)]


def _strip_comments(src: str) -> str:
    """Drop comments and docstrings so prose about the old bug is not mistaken for code."""
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


# Every door onto svc.refresh(), and the metering call each one must make first.
PAID_REFRESH_DOORS = [
    pytest.param(RESEARCH_ROUTES, '@router.post("/track/{tracked_job_id}/refresh"',
                 "debit_credits", id="session-jwt (the app's Refresh button)"),
    pytest.param(TRACKING_ROUTES, '@router.post("/{tracking_id}/refresh"',
                 "debit_credits", id="partner api-key"),
    pytest.param(RESEARCH_ROUTES, '@router.post("/cron-refresh"',
                 "charge_cron", id="cron sweep"),
]


@pytest.mark.parametrize("path, decorator, meter_fn", PAID_REFRESH_DOORS)
def test_paid_refresh_door_meters_before_it_spends(path: Path, decorator: str, meter_fn: str):
    body = _strip_comments(_route_body(_read(path), decorator))

    assert meter_fn in body, (
        f"{path.name} {decorator} calls a paid refresh without calling {meter_fn}(). "
        "Every door onto svc.refresh() must meter — the unmetered one is always the one "
        "users find."
    )

    meter_at = body.index(meter_fn)
    refresh_at = body.index("svc.refresh(") if "svc.refresh(" in body else body.index(".refresh(")
    assert meter_at < refresh_at, (
        f"{path.name} {decorator} calls {meter_fn}() AFTER the refresh. Invariant 10 requires "
        "the debit BEFORE the upstream call: charging afterwards means a caller with no credits "
        "still spent the operator's DataForSEO/Firecrawl/Anthropic budget."
    )


def test_session_refresh_refunds_when_the_work_could_not_be_delivered():
    """A debit with no refund path is a standing charge for nothing (see cron-billing's own
    docstring on the same hazard)."""
    body = _strip_comments(_route_body(_read(RESEARCH_ROUTES),
                                       '@router.post("/track/{tracked_job_id}/refresh"'))
    assert "refund_credits" in body, "the session refresh debits but never refunds"
    assert "except" in body, "an exception mid-refresh must refund, not keep the credit"


def test_session_refresh_refuses_rather_than_falling_open_without_a_payer():
    """`if user_id and not debit(...)` reads as metering and behaves as a free pass whenever the
    identity is missing. The session door must refuse instead."""
    body = _strip_comments(_route_body(_read(RESEARCH_ROUTES),
                                       '@router.post("/track/{tracked_job_id}/refresh"'))
    assert not re.search(r"if\s+user_id\s+and\s+not\s+costs\.debit_credits", body), (
        "the debit is guarded by `if user_id and ...` — a missing sub silently skips the charge"
    )
    assert "401" in body, "a request with no resolvable payer must be refused, not served free"


def test_both_refresh_doors_agree_on_the_price():
    """One price for one operation. Two doors quoting different numbers is the same
    derived-copy drift that money quantities suffer from."""
    for path, decorator, _ in [(p.values[0], p.values[1], None) for p in PAID_REFRESH_DOORS[:2]]:
        body = _strip_comments(_route_body(_read(path), decorator))
        assert 'JOB_OP_CREDIT_COST.get("refresh"' in body, (
            f"{path.name} hard-codes a refresh price instead of reading JOB_OP_CREDIT_COST"
        )
