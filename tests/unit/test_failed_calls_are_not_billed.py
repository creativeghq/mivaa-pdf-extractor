"""
A provider call that failed must not appear as money we spent.

`log_external_call` wrote `billed = raw_cost_usd * markup` whatever `success` said, and put
`success` only into `metadata`. Every flat-rate provider therefore booked its per-call price for
calls the provider refused. Measured on live data 2026-08-30, over seven days:

    dataforseo-labs-related-keywords   53 failed calls   $0.954 billed
    sonar (Perplexity, HTTP 401)       35 failed calls   $0.263 billed
    dataforseo-serp-google-organic      4 failed calls   $0.004 billed
    firecrawl-v2                        1 failed call    $0.003 billed

Token-priced models never had the bug — their cost comes from `response.usage` and a failed call
returns no tokens, so the arithmetic already gave zero. That is exactly why this looked like four
provider-specific quirks and was one line in the shared writer.

The failure mode is the dangerous kind: a plausible number. Nothing errors, nothing is missing,
and every cost view silently includes spend that never happened.

MIVAA's CI installs pytest and no application dependencies, so `cost_logger.py` (which imports a
Supabase client at module scope) cannot be imported here. `cost_accounting.py` imports nothing and
is loaded by path, so this exercises the REAL derivation rather than asserting on source text —
and a second, source-level check makes sure the writer still calls it.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACCOUNTING = ROOT / "app" / "modules" / "_core" / "cost_accounting.py"
WRITER = ROOT / "app" / "modules" / "_core" / "cost_logger.py"


def _load():
    spec = importlib.util.spec_from_file_location("cost_accounting", ACCOUNTING)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_failed_call_costs_nothing_and_says_why():
    m = _load()
    raw, billed, reason = m.settle_call_cost(0.005, 1.5, success=False)
    assert raw == 0.0, "a call the provider refused is not money we spent"
    assert billed == 0.0
    assert reason == "call_failed", "a zero cost must be distinguishable from a free success"


def test_a_successful_call_is_priced_exactly_as_before():
    m = _load()
    raw, billed, reason = m.settle_call_cost(0.005, 1.5, success=True)
    assert raw == 0.005
    assert billed == 0.0075
    assert reason is None, "a success must not be labelled unbilled"


def test_the_intended_price_survives_a_failure():
    # Zeroing the cost must not destroy the difference between "this provider is free" and
    # "this provider charges real money and refused us 35 times".
    m = _load()
    assert m.intended_cost_usd(0.005) == 0.005
    assert m.intended_cost_usd(0) == 0.0


def test_the_flat_rate_offenders_are_all_covered_by_one_writer():
    # sonar / firecrawl / dataforseo all route through log_external_call, which is why this is a
    # single fix. If a module grows its own ai_usage_logs insert, it escapes this rule.
    writer = WRITER.read_text(encoding="utf-8")
    assert "settle_call_cost(" in writer, "the writer no longer asks whether the call succeeded"
    assert '"unbilled_reason": unbilled_reason' in writer, "the row no longer says why it is zero"
    assert "would_have_cost_usd" in writer, "the intended price is being discarded on failure"
    # The old unconditional arithmetic must not come back.
    assert "float(raw_cost_usd) * float(markup_multiplier)" not in writer, (
        "billed cost is being computed without consulting `success` again"
    )
