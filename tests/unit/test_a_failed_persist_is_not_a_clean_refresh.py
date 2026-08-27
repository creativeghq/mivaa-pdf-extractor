"""A refresh that could not persist must not report success (mivaa#19 M6-3/M6-4/M6-5).

The shape, in one sentence: the write failed, and the code then advanced every piece of
state that means "this worked".

    tracked_queries    the history insert raised -> `rows = []` -> the row was updated
                       with `last_error: None`, `first_refresh_verified: True` and a
                       full set of NULL `current_*` columns, and the call returned
                       `status: "refreshed"`.

So a database outage CLEARED the previous error, wiped a cached price that was still the
best answer anyone had, marked the row verified, and told cron there was nothing to
retry. The dashboard showed a confident empty. Nothing raised, nothing logged above
warning, and `ops.silent_zero` cannot see it — the metric it would look at is the one
that got nulled.

M6-3 is the same defect one layer over: the gold `current_*` cache was summarised from
the IN-MEMORY rows rather than the committed ones, so a partial write produced a
confident summary of silver rows nobody can read back — sitting next to counts that ARE
read from the database, so the row disagreed with itself.

M6-5 is the same shape in the failure ledger: a Claude call that timed out AFTER
Anthropic accepted and billed it was written as `cost=0.0` with nothing marking it, so
the spend was permanently indistinguishable from a free no-op.

Static analysis, so these pin the SHAPE. Every case was watched to fail against the
pre-fix source.
"""

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

QUERIES = APP / "services" / "integrations" / "tracked_queries_service.py"
MENTIONS = APP / "services" / "integrations" / "tracked_mentions_service.py"
CLAUDE_HELPER = APP / "services" / "core" / "claude_helper.py"


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


# ───────────────────────────────────────────────────────────────────────────
# M6-4 — the failure has to reach the row, the caller and the cron
# ───────────────────────────────────────────────────────────────────────────

def test_a_failed_history_insert_is_recorded_rather_than_cleared():
    body = _strip_comments(_source_of(_read(QUERIES), "_persist_refresh"))

    assert "persist_error" in body, "the persist outcome is no longer tracked at all"
    assert '"last_error": persist_error' in body, (
        'the update writes `last_error: None` again — a failed refresh CLEARS the '
        "previous error, which is worse than not recording the new one"
    )
    assert re.search(r"if persist_error is None:\s*\n\s*update_payload\[.first_refresh_verified.\]", body), (
        "`first_refresh_verified` is set unconditionally again, so a run that persisted "
        "nothing marks the row verified"
    )


def test_the_outcome_reported_to_the_caller_distinguishes_the_two():
    """Cron decides whether to retry from this. `refreshed` is a claim that the run is
    durably readable."""
    body = _strip_comments(_source_of(_read(QUERIES), "_persist_refresh"))
    assert re.search(r'"status": "refreshed" if persist_error is None else "error"', body), (
        "the refresh reports `refreshed` regardless of whether anything persisted"
    )


def test_persist_error_is_bound_before_every_read():
    """It is assigned inside `if hits:`, and a run that found nothing never enters that
    branch — so binding it there would raise UnboundLocalError on exactly the quiet path
    this test exists to protect."""
    src = _read(QUERIES)
    fn = _node(src, "_persist_refresh")
    binds = [n.lineno for n in ast.walk(fn)
             if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "persist_error"]
    binds += [n.lineno for n in ast.walk(fn)
              if isinstance(n, ast.Assign)
              and any(getattr(t, "id", "") == "persist_error" for t in n.targets)]
    reads = [n.lineno for n in ast.walk(fn)
             if isinstance(n, ast.Name) and n.id == "persist_error" and isinstance(n.ctx, ast.Load)]
    assert binds and reads
    assert min(binds) < min(reads), (
        "every binding of persist_error now sits after its first read — the empty-result "
        "path raises UnboundLocalError"
    )


# ───────────────────────────────────────────────────────────────────────────
# M6-3 — the gold cache summarises what was COMMITTED, and only when there is some
# ───────────────────────────────────────────────────────────────────────────

def test_the_cache_is_built_from_what_the_database_accepted():
    body = _strip_comments(_source_of(_read(QUERIES), "_persist_refresh"))
    assert "insert_resp" in body and "committed" in body, (
        "the insert response is discarded again, so `current_*` summarises the rows we "
        "MEANT to write rather than the ones that landed"
    )


def test_an_empty_refresh_does_not_wipe_the_last_good_price():
    """`_select_cheapest([])` is None, and the old form turned that into a full set of
    NULLs. One quiet refresh then erased a price that was still the best answer — while
    `_refresh_url_only`'s own comment promised the opposite."""
    body = _strip_comments(_source_of(_read(QUERIES), "_persist_refresh"))
    assert re.search(r"if cheapest:\s*\n\s*update_payload\.update\(", body), (
        "the current_* columns are written unconditionally again"
    )
    for column in ("current_price", "current_currency", "current_price_verified"):
        assert f'"{column}": cheapest.get' in body or f'"{column}": bool(cheapest' in body, (
            f"{column} is no longer guarded by `if cheapest`"
        )


def test_the_mention_cache_is_not_stamped_from_rows_that_never_landed():
    """`current_sentiment_avg` and `current_top_outlets` are computed in memory, and the
    counts beside them are read from the database — so stamping the first pair after a
    failed insert makes the row disagree with itself."""
    src = _read(MENTIONS)
    node = _node(src, "_stamp_refresh")
    names = [a.arg for a in node.args.args] + [a.arg for a in node.args.kwonlyargs]
    assert "history_persisted" in names, (
        "_stamp_refresh no longer knows whether the history it summarises was written"
    )

    body = _strip_comments(_source_of(src, "_stamp_refresh"))
    assert re.search(r"if history_persisted:", body), "the flag is accepted and ignored"
    for derived in ("current_sentiment_avg", "current_top_outlets"):
        at = body.index(derived)
        assert body.index("if history_persisted:") < at, (
            f"{derived} is stamped before the persisted check, so a failed insert still "
            "overwrites it"
        )


def test_the_caller_passes_the_persisted_flag_through():
    """A parameter with a safe default that nobody supplies is a fix that never runs."""
    body = _strip_comments(_read(MENTIONS))
    assert "history_persisted=history_persisted" in body, (
        "refresh() calls _stamp_refresh without telling it whether the insert worked"
    )


# ───────────────────────────────────────────────────────────────────────────
# M6-5 — an unknown cost must not be recorded as a zero one
# ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "fn", ["_log_failed_claude_call_async", "_log_failed_claude_call_sync"]
)
def test_a_failed_billable_call_is_marked_not_just_zeroed(fn: str):
    """`cost=0.0` is right — the cost is not recoverable client-side — but it is also
    what a free no-op looks like. `unbilled_reason` is the column that already exists to
    separate them: NULL means billed, anything else names why it was not."""
    body = _strip_comments(_source_of(_read(CLAUDE_HELPER), fn))
    assert "unbilled_reason=" in body, (
        f"{fn} writes a zero-cost row with no marker, so a call that Anthropic accepted "
        "and billed is indistinguishable from one that never cost anything"
    )
    assert "billable_attempt_failed" in body
