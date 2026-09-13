"""
Guard: a provider refusing our CREDENTIAL is never rendered as zero results (#416).

WHY THIS EXISTS
---------------
Perplexity answered 401 on every call for six weeks -- 198 a week across two subsystems that
share one key -- and both rendered the answer as "no mentions" / "no jobs". `last_success_at` was
NULL on both. Every artifact was well-formed: the call was made, it was logged as a failure, and
the function returned an empty list that the fan-out counted as a source that ran and found
nothing.

Two things are pinned here:

1. The POLICY: only refusals the credential causes may open the breaker. Opening on a timeout or
   a 500 turns a provider's bad minute into a self-inflicted outage that outlives it.
2. The SHAPE: `call_agent` consults the breaker before spending, and the two callers say WHY they
   have nothing -- mention search through a status on its result, job search by raising, because
   its contract is a bare list with nowhere to carry one.

Loaded BY PATH and stdlib-only: MIVAA's CI installs pytest alone, so a module-level provider
import would put this out of its own reach.
"""

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INTEG = ROOT / "app" / "services" / "integrations"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, INTEG / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pb = _load("_pb_under_test", "provider_breaker.py")

_DQ = chr(34) * 3
_SQ = chr(39) * 3


def _src(rel):
    """Source with comments and docstrings blanked -- a comment must not satisfy a check."""
    text = (INTEG / rel).read_text(encoding="utf-8")
    for fence in (_DQ, _SQ):
        text = re.sub(re.escape(fence) + r"[\s\S]*?" + re.escape(fence), '""', text)
    return re.sub(r"(?m)#.*$", "", text)


class TestOnlyTheCredentialOpensIt:
    def test_auth_statuses_are_credential_failures(self):
        assert pb.classify_provider_failure(401, "") == "auth"
        assert pb.classify_provider_failure(403, "") == "auth"
        assert pb.classify_provider_failure(402, "") == "quota"

    def test_a_bad_minute_is_not_a_dead_key(self):
        for status in (500, 502, 503, 504, 408, None):
            assert pb.classify_provider_failure(status, "upstream exploded") is None, status

    def test_429_is_read_from_the_body_not_the_status(self):
        assert pb.classify_provider_failure(429, "slow down") is None
        assert pb.classify_provider_failure(
            429, "You exceeded your current quota, please check your plan and billing details."
        ) == "quota"


class TestTheVerdict:
    def test_an_unreadable_state_still_calls(self):
        # Failing closed here would let one unreachable table silence every provider at once.
        assert pb.verdict_from_state(None).should_call is True

    def test_open_refuses_and_carries_the_providers_own_message(self):
        v = pb.verdict_from_state(
            {"open": True, "consecutive_auth_failures": 5, "last_error": "401 Unauthorized"}
        )
        assert v.should_call is False
        assert v.refused is True
        assert v.status == "collector_failed"
        assert "401 Unauthorized" in v.reason

    def test_half_open_lets_exactly_one_through(self):
        v = pb.verdict_from_state({"open": True, "probe": True, "consecutive_auth_failures": 9})
        assert v.should_call is True and v.is_probe is True


class TestTheCallersSayWhy:
    def test_call_agent_asks_the_breaker_before_spending(self):
        src = _src("perplexity_agent_client.py")
        assert "breaker_verdict" in src, "call_agent spends without asking whether the key works"
        assert src.index("breaker_verdict") < src.index("client.post"), (
            "the breaker is consulted after the request -- it has to stop the spend, not report it"
        )
        assert "record_outcome" in src, "nothing ever closes the breaker again"

    def test_mention_search_states_a_reason_instead_of_an_empty_list(self):
        src = _src("mention_search_service.py")
        assert "def source_failed" in src
        assert "source_not_connected" in src
        assert "errors[name]" in src
        assert 'if status and status != "ok"' in src

    def test_job_search_raises_rather_than_returning_no_jobs(self):
        src = _src("job_search_service.py")
        assert "ProviderRefused" in src, (
            "search_via_perplexity returns a bare list, so a credential refusal that returns [] "
            "is counted by the fan-out as a source that ran and found nothing"
        )
        assert "is_credential_refusal" in src
