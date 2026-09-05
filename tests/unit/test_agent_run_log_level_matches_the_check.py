"""The job-research run log writes a level the DB CHECK accepts (MIVAA-5JV).

`agent_run_logs_level_check` allows `debug | info | warn | error` — the edge writer's
vocabulary. Python spells the third one `warning`, every caller here did as well, and
`append_log` passed it through verbatim: each warning-level row was refused with 23514 and
the refusal was logged at DEBUG, so the run's audit trail silently lost exactly the lines
that said something had gone wrong.

Source-based, like the rest of this suite: MIVAA's CI installs pytest and no application
dependencies, so the module cannot be imported here. The normaliser itself is pure and is
exercised by loading it out of the file.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WRITER = ROOT / "app" / "services" / "integrations" / "job_agent_runs.py"

#: What the CHECK constraint accepts, verbatim (2026-09-05).
CHECK_LEVELS = {"debug", "info", "warn", "error"}


def _writer_source() -> str:
    return WRITER.read_text(encoding="utf-8")


def _load_normaliser():
    """Execute only the level-normalising block, which has no imports."""
    src = _writer_source()
    start = src.index("_RUN_LOG_LEVELS = frozenset(")
    end = src.index("def append_log(")
    ns: dict = {}
    exec(src[start:end], ns)  # noqa: S102 - our own source, no I/O
    return ns


def test_the_insert_normalises_the_level_instead_of_passing_it_through():
    src = _writer_source()
    assert '"level": normalize_run_log_level(level)' in src, (
        "append_log must run the level through normalize_run_log_level before the insert; "
        "a raw `level` is how `warning` reached a CHECK that only knows `warn`"
    )
    assert '"level": level,' not in src


def test_the_normaliser_only_ever_answers_with_a_check_value():
    ns = _load_normaliser()
    normalise = ns["normalize_run_log_level"]
    assert ns["_RUN_LOG_LEVELS"] == frozenset(CHECK_LEVELS)
    for given in ["debug", "info", "warn", "warning", "WARNING", "error", "critical", "fatal", "", "nonsense", None]:
        assert normalise(given) in CHECK_LEVELS, given
    assert normalise("warning") == "warn"
    assert normalise("critical") == "error"
    assert normalise("nonsense") == "info"


def test_a_refused_row_is_not_hidden_at_debug():
    src = _writer_source()
    body = src[src.index("def append_log("):src.index("def complete_run(")]
    assert "logger.warning(" in body and "logger.debug(" not in body, (
        "a lost run log is a lost audit line; logging the refusal at DEBUG is how MIVAA-5JV "
        "went unnoticed"
    )


def test_every_caller_passes_a_level_the_normaliser_understands():
    known = CHECK_LEVELS | {"warning", "critical", "fatal"}
    offenders = []
    for path in (ROOT / "app").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"append_log\((?:[^)]|\n)*?level=\"([a-z]+)\"", text):
            if m.group(1) not in known:
                offenders.append(f"{path.relative_to(ROOT)}: {m.group(1)}")
    assert not offenders, offenders
