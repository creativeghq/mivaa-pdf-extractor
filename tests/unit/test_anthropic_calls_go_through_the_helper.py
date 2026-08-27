"""Files calling Anthropic directly may only DECREASE (mivaa#33 item 2).

`claude_helper` is the correct path and the reason is not style. It:

  * recovers the billable principal from `job_id` when `user_id` is absent, so the
    spend is attributed instead of landing on nobody
  * logs failures, including the `billable_attempt_failed` marker added in 6ef86a4 —
    a raw httpx POST that times out AFTER Anthropic accepted and billed the request
    leaves no trace at all
  * is the one place `tools` / `tool_choice` can be forced from (`claude_tool_call`
    builds on it)

26 of 41 Anthropic-calling files use it. The rest hit `api.anthropic.com` or
`messages.create` themselves, so they reach the `if user_id:` branch with `None`, skip
the debit, and write no UNBILLED marker.

WHY A RATCHET
-------------
Fifteen files cannot honestly be migrated in one change — each has its own payload
shape, retry policy and failure semantics — and a guard that fails from the day it is
written gets deleted rather than satisfied. Same contract as
`.github/ruff-baseline.json` and the #32 parser ratchet: a NEW bypass fails the build
immediately, the existing debt is counted and visible, and the number may only fall.

Fix findings; do not edit the baseline upward.

ON THE AUDIT'S LIST
-------------------
It names 15 files. Measured against the tree, the membership differs in two places and
both matter for anyone working the list:

  * `anthropic_error_reporter` is NOT a bypass. Its `messages.create(...)` is inside the
    module DOCSTRING, showing callers how to use the reporter. It never calls Anthropic —
    it reports failures to Sentry. This sweep strips docstrings and comments, which is
    why it does not appear.
  * `mention_identity_service` IS a bypass and is not on the audit's list.
"""

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
BASELINE = ROOT / ".github" / "anthropic-bypass-baseline.json"

SKIP_DIRS = {"__pycache__", ".venv", "venv", "node_modules"}

#: The helper itself, and the module that owns the shared provider client, are where
#: these calls are SUPPOSED to live.
ALLOWED = {
    "app/services/core/claude_helper.py",
    "app/services/core/ai_client_service.py",
}

#: A direct call to Anthropic, bypassing the tracked helper.
_DIRECT = ("api.anthropic.com", "messages.create")


def _strip_comments(src: str) -> str:
    """Docstrings and comments are PROSE. A usage example in a docstring is not a call,
    and counting it as one is how a sweep produces a finding nobody can act on."""
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def sweep():
    """Relative paths that call Anthropic directly without going through the helper."""
    offenders = []
    for path in sorted(APP.rglob("*.py")):
        if any(d in path.parts for d in SKIP_DIRS):
            continue
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        if rel in ALLOWED:
            continue
        try:
            src = _strip_comments(path.read_text(encoding="utf-8"))
        except OSError:  # pragma: no cover
            continue
        if not any(marker in src for marker in _DIRECT):
            continue
        # A file that ALSO routes through the helper still counts: the point is that
        # this particular call does not, and a half-migrated file is the easiest place
        # for the next one to be added.
        offenders.append(rel)
    return sorted(offenders)


def _baseline() -> dict:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_no_new_file_calls_anthropic_directly():
    offenders = sweep()
    recorded = _baseline()
    known = set(recorded["known"])
    new = sorted(set(offenders) - known)
    assert not new, (
        "these files call Anthropic directly and are not on the baseline:\n  "
        + "\n  ".join(new)
        + "\n\nUse `claude_helper.tracked_claude_call_async` (or "
          "`claude_tool_call.call_with_tool` for a structured reply). A raw POST is "
          "invisible to cost attribution and leaves no record when it fails."
    )
    assert len(offenders) <= recorded["bypassing_files"], (
        f"direct-call files rose from {recorded['bypassing_files']} to {len(offenders)}"
    )


def test_the_baseline_comes_down_when_the_work_is_done():
    """A baseline nobody lowers has stopped meaning anything."""
    offenders = sweep()
    recorded = _baseline()["bypassing_files"]
    assert len(offenders) >= recorded - 2, (
        f"the count dropped to {len(offenders)} against a baseline of {recorded} — "
        "lower the baseline in the same change that fixed them, so the next regression "
        "is caught against the new floor"
    )


def test_the_helper_is_not_itself_on_the_list():
    """If `claude_helper` ever stops calling Anthropic, everything below it is calling
    something else and this whole guard is measuring the wrong thing."""
    helper = (APP / "services" / "core" / "claude_helper.py").read_text(encoding="utf-8")
    assert any(m in _strip_comments(helper) for m in _DIRECT), (
        "claude_helper no longer calls Anthropic — this sweep's premise is gone"
    )


def test_a_docstring_example_is_not_counted_as_a_call():
    """`anthropic_error_reporter` shows `await self.client.messages.create(...)` in its
    module docstring to teach callers how to use it. It never calls Anthropic. The
    audit's list counts it; this sweep must not."""
    reporter = "app/services/core/anthropic_error_reporter.py"
    assert reporter not in sweep(), (
        "the sweep is counting a docstring usage example as a real call — that is a "
        "finding nobody can act on"
    )
    raw = (ROOT / reporter).read_text(encoding="utf-8")
    assert "messages.create" in raw, (
        "the docstring example is gone, so this case no longer proves anything — "
        "point it at another file that documents a call it does not make, or delete it"
    )
