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


# ═══════════════════════════════════════════════════════════════════════════
# The monitoring modules — enumerated, not listed
# ═══════════════════════════════════════════════════════════════════════════
#
# Everything above pins THREE doors by hand. That is why audit #18 M5-3 found
# thirteen more entering Perplexity / DataForSEO / Firecrawl / LLM work with no
# debit at all: they were outside a test whose reassuring name ("paid route
# metering") is exactly why nobody looked.
#
# So this half does not name doors. It finds every route in the two monitoring
# modules that reaches a paid operation and asserts each one meters — which means
# a NEW paid door fails this test on the day it is written, without anyone
# remembering to add it here.

MONITORING_MODULES = [
    ROOT / "app" / "api" / "price_monitoring_routes.py",
    ROOT / "app" / "api" / "mention_monitoring_routes.py",
    # Added by #23 M10-4. These eight doors reached Claude and Voyage with no debit at
    # all — and were outside this file for the same reason the thirteen were: nobody
    # thought of "AI services" as a monitoring module, and the test is named after the
    # modules it happened to cover rather than the property it checks.
    ROOT / "app" / "api" / "ai_services_routes.py",
]

#: Calls that spend money at a provider. A route reaching one of these is a paid door.
PAID_CALLS = (
    ".refresh(",
    ".reverify(",
    ".probe(",
    ".generate(",
    ".search_prices(",
    ".find_or_create_for_product(",
    ".add_url_only(",
    # #23 M10-4 — every provider call behind /api/v1/ai-services.
    ".classify_content(",
    ".classify_batch(",
    ".detect_boundaries(",
    ".group_chunks_by_product(",
    ".validate_product(",
    ".validate_critical_extraction(",
)

#: `find_or_create_for_product` is the one call above that is only SOMETIMES paid --
#: enrolling a subject is free, running its first discovery sweep is not. Enrolment-only
#: calls say so explicitly, so they are excised before a route is judged. Without this
#: the sweep flags /start (enrol, no sweep) and reads the free auto-enrol at the top of
#: mention's refresh/probe routes as spend that happens before the debit.
_FREE_ENROLMENT = re.compile(
    r"\.find_or_create_for_product\((?:[^()]|\([^()]*\))*run_first_refresh=False(?:[^()]|\([^()]*\))*\)"
)

#: The two ways a door may meter: a user wallet through the shared wrapper, or the
#: cron meter for scheduled callers.
METERS = ("metered_door(", "charge_cron(")


def _routes(src: str):
    """(decorator_line, body) for every @router.<verb> handler in the module."""
    for m in re.finditer(r"^@router\.(get|post|put|patch|delete)\(", src, re.MULTILINE):
        nxt = src.find("\n@router.", m.end())
        yield src[m.start(): m.start() + src[m.start():].index("\n")], \
            src[m.start(): nxt if nxt != -1 else len(src)]


def _paid_doors():
    for path in MONITORING_MODULES:
        src = _read(path)
        for decorator, body in _routes(src):
            stripped = _FREE_ENROLMENT.sub("", _strip_comments(body))
            if any(call in stripped for call in PAID_CALLS):
                yield pytest.param(path, decorator, stripped,
                                   id=f"{path.stem.split('_')[0]}:{decorator[8:60]}")


def test_the_ai_service_doors_price_by_the_work_not_by_the_call():
    """A flat price on `/classify-batch` makes the 50-item call the cheap way to buy 50
    Claude calls. The cap from M10-2 bounds the amplification; only per-item pricing
    removes the incentive to use it."""
    src = _strip_comments(_read(ROOT / "app" / "api" / "ai_services_routes.py"))
    assert "CREDIT_PER_CLASSIFY * len(request.contents)" in src, (
        "classify_batch no longer prices per item"
    )
    assert src.count("_chunk_cost(len(request.chunks))") == 3, (
        "a chunk-driven door stopped pricing by its chunk count"
    )


@pytest.mark.parametrize("path, decorator, body", list(_paid_doors()))
def test_every_monitoring_door_that_spends_money_meters_first(path, decorator, body):
    meter_at = min((body.index(m) for m in METERS if m in body), default=-1)
    assert meter_at != -1, (
        f"{path.name} {decorator} reaches a paid provider call with no metering.\n"
        "Wrap the call in `metered_door(...)` (user-triggered) or gate it on "
        "`charge_cron(...)` (scheduled). Invariant 10: debit BEFORE the upstream call."
    )

    spend_at = min(body.index(c) for c in PAID_CALLS if c in body)
    assert meter_at < spend_at, (
        f"{path.name} {decorator} meters AFTER it spends. A caller with no credits still "
        "burned the operator's provider budget."
    )


def test_the_paid_door_wrapper_refunds_at_most_once():
    """The wrapper is what makes refunds idempotent — without it, a door that grows a
    second refund branch hands the credit back twice, and an over-refund is as invisible
    as an under-charge (#18 M5-3)."""
    src = _read(ROOT / "app" / "utils" / "paid_door.py")
    body = _strip_comments(src)
    assert "_refunded" in body, "PaidWork no longer tracks whether it already refunded"
    assert re.search(r"if self\._refunded[^\n]*:\s*\n\s*return", body), (
        "PaidWork.refund() no longer short-circuits on a second call"
    )


def test_cron_metering_fails_closed_on_a_charge_failure():
    """A cron is the highest-volume caller of these endpoints, so failing open converts a
    billing outage into unbounded free provider spend across every scheduled run
    (#18 M5-4). Invariant 10 is explicit: on debit failure, do not perform the work."""
    body = _strip_comments(_read(ROOT / "app" / "services" / "integrations" / "cron_billing.py"))
    assert "return True  # nothing returned -> fail open" not in body
    assert not re.search(r"except Exception[^\n]*\n(?:[^\n]*\n){0,3}?\s*return True", body), (
        "charge_cron returns True from its exception handler again — a failed charge "
        "must skip the unit, not spend on the payer's behalf"
    )
    assert "_record_unmetered" in body, (
        "an unmetered scheduled unit leaves no durable record, so it is indistinguishable "
        "from a metered one by the time anyone looks (#17 M4-2)"
    )
