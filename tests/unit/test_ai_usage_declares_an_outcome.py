"""
Guard: every writer into `ai_usage_logs` says whether the call SUCCEEDED.

WHY THIS EXISTS
---------------
`ops.silent_zero_provider` — the probe that catches "this provider is refusing every
call we pay for" — judges a provider only on rows whose `metadata` carries a `success`
key. That is the correct rule and its own comment explains it: a row that never claimed
to succeed is not evidence that it failed, and counting silent rows as failures would
flag every logger that simply does not record an outcome.

The rule is only useful if the writers hold up their end. They did not.

Measured 2026-08-23 over seven days of `ai_usage_logs`:

    provider      calls   declare an outcome
    voyage          510                    0
    anthropic       596                   12
    perplexity       30                   30   <- the only one fully watched
    google/apollo/
    firecrawl/…      20                    0

So roughly 1,100 calls a week sat outside the probe's field of view, and the single
provider it could see is the one it correctly caught (Perplexity, 401 on every call).

This is not theoretical. On 2026-08-22 the platform's Anthropic account hit a zero
balance and every agent, vision and classifier call began returning 400. The probe said
nothing and could not have. The outage was found by a person noticing the agent replying
with an error string.

It also produces confidently wrong readings on the way past: querying
`metadata->>'success' = 'true'` across those rows returns a plausible low percentage that
measures the ABSENCE OF A FIELD, not failure.

WHAT THIS ASSERTS
-----------------
Every `ai_usage_logs` insert in `app/` carries a `success` key in its metadata. Source
based, so it runs in this repo's pytest-only CI.
"""

import io
import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_APP = _ROOT / "app"

# Reads are not writes. A file that only SELECTs from the table has no outcome to declare.
_INSERT_RE = re.compile(r'table\(\s*["\']ai_usage_logs["\']\s*\)\s*\.\s*insert', re.S)


def _writers():
    """Files that INSERT into ai_usage_logs, with their source."""
    out = []
    for path in sorted(_APP.rglob("*.py")):
        src = io.open(path, encoding="utf-8", errors="ignore").read()
        if _INSERT_RE.search(src):
            out.append((path.relative_to(_ROOT).as_posix(), src))
    return out


WRITERS = _writers()


def test_the_scan_finds_the_writers_at_all():
    """A guard that matched nothing would pass forever."""
    assert len(WRITERS) >= 4, (
        f"only {len(WRITERS)} ai_usage_logs writer(s) found — the insert pattern probably "
        f"changed shape, and this guard is now watching nothing"
    )


@pytest.mark.parametrize("rel,src", WRITERS, ids=[w[0] for w in WRITERS])
def test_every_writer_declares_an_outcome(rel: str, src: str):
    """The row must carry `success`, so the provider probe can judge it.

    Deliberately a check for the KEY, not for a particular value or shape. A writer that
    only ever logs successes may hardcode `"success": True` — that is a claim, and a claim
    is what the probe needs. Silence is the only wrong answer.
    """
    assert '"success"' in src or "'success'" in src, (
        f"{rel} inserts into ai_usage_logs without a `success` key in metadata. "
        f"`ops.silent_zero_provider` reads that key and skips any row lacking it, so every "
        f"call this file logs would be invisible to the probe that exists to notice a "
        f"provider refusing all of them — which is exactly how the 2026-08-22 Anthropic "
        f"outage went unreported."
    )


def test_the_shared_logger_derives_success_rather_than_taking_it_on_faith():
    """`ai_call_logger` mirrors most of the platform's calls, so it matters most.

    It already receives `error_message`; `success` is derived from it in one place rather
    than passed as a second parameter that could disagree with the first.
    """
    src = io.open(_APP / "services" / "core" / "ai_call_logger.py", encoding="utf-8").read()
    assert '"success": error_message is None' in src, (
        "ai_call_logger must derive `success` from `error_message`, not accept it "
        "separately — two fields describing one outcome will eventually disagree"
    )
    assert '"error": error_message' in src, (
        "the error text belongs on the row too: a probe can say WHICH provider is failing, "
        "and only the row can say why"
    )
