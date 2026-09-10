"""Free-form model-reply parsers may only DECREASE (mivaa#32).

34 parsers across 22 files ask a model for JSON in a prompt and then repair the reply
when it is not JSON — 15 by stripping markdown fences. The fence-strip is the tell: it
is an admission, written by someone who had already watched the free-form contract
break.

WHY A RATCHET AND NOT A ZERO
----------------------------
Because 34 cannot honestly be fixed in one change, and a guard that fails from the day
it is written gets deleted rather than satisfied. This is the same contract as
`.github/ruff-baseline.json` and the platform's `edge-typecheck-baseline.json`: the
recorded count may only fall. A NEW parser fails the build immediately; the existing
debt is visible, counted, and shrinking.

Fix findings; do not edit the baseline upward.

WHAT COUNTS
-----------
A function that mentions Claude/Anthropic, parses a reply as free-form text
(`json.loads`, a JSON regex, `content[0].text`), and does NOT set `tool_choice` in the
same function. Functions that force the tool call are excluded — that is the fix.

INVARIANT 9 IS NARROWER THAN THIS SWEEP, AND SAYING SO MATTERS
--------------------------------------------------------------
The platform rule mandates forced tool-calling for "a classifier whose verdict drives a
DB write or alert". A parser feeding a search ranking is a robustness concern; one
feeding an INSERT is the invariant. Both are counted here, because both benefit and the
sweep cannot tell them apart mechanically — but the count is not a count of invariant
violations, and reading it as one would overstate the case.
"""

import ast
import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
BASELINE = ROOT / ".github" / "freeform-parser-baseline.json"

SKIP_DIRS = {"__pycache__", ".venv", "venv", "node_modules"}

#: A file is in scope only if it talks to Claude at all.
_PROVIDER = re.compile(r"claude|anthropic", re.IGNORECASE)

#: Signs that a function is parsing a model reply as text.
_PARSES = (
    "json.loads",
    "content[0].text",
    ".text.strip()",
)

#: Markdown-fence repair — the strongest single signal, counted separately.
_FENCE = ("```json", '"```"', "```")

#: The TRANSPORT, which is not a parser: it decodes Anthropic's own wire protocol, including the
#: `input_json_delta` fragments a FORCED tool call's input arrives in.
_TRANSPORT = {
    "app/services/core/claude_helper.py::tracked_claude_stream_async",
    # Its inner half. Both forward whatever the caller forces via `**extra`; neither decides
    # anything from the reply, so neither is a parser.
    "app/services/core/claude_helper.py::_stream_anthropic_async",
}

#: A body carrying one of these has forced the tool and cannot receive prose.
_FORCED = ("tool_choice", "call_with_tool")


def _load_blank_comments():
    """Load the comment blanker by path — this test imports no app module."""
    spec = importlib.util.spec_from_file_location(
        "comment_budget_for_replies", ROOT / "scripts" / "comment_budget.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.blank_comments


_blank_comments = _load_blank_comments()


def _files():
    return sorted(p for p in APP.rglob("*.py") if not any(d in p.parts for d in SKIP_DIRS))


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def sweep():
    """(all_parsers, fence_strippers) as sorted 'path::function' lists."""
    parsers, fences = [], []
    for path in _files():
        try:
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except SyntaxError:
            continue
        # The sweep reads CODE. Against the raw text, a function was exempted because a comment
        # happened to say `tool_choice`, and `classify_images` — which forces the tool through
        # call_with_tool — was skipped for the words in a comment rather than for what it does.
        code = _blank_comments(src)
        if not _PROVIDER.search(code):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = ast.get_source_segment(code, node) or ""
            if any(marker in body for marker in _FORCED):
                continue  # forced — this is the fix, not the defect
            if not any(marker in body for marker in _PARSES):
                continue
            if not _PROVIDER.search(body):
                continue
            key = f"{_rel(path)}::{node.name}"
            if key in _TRANSPORT:
                continue
            parsers.append(key)
            if any(f in body for f in _FENCE):
                fences.append(key)
    return sorted(set(parsers)), sorted(set(fences))


def _baseline() -> dict:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_the_free_form_parser_count_never_rises():
    """A NEW parser fails here on the day it is written. Route it through
    `claude_tool_call.call_with_tool` instead: forced `tool_choice` means the model
    cannot return prose, so there is nothing to repair and an absent tool block is a
    typed, retryable error rather than a silent None."""
    parsers, _ = sweep()
    recorded = _baseline()["free_form_parsers"]
    assert len(parsers) <= recorded, (
        f"free-form model-reply parsers rose from {recorded} to {len(parsers)}.\n"
        "New:\n  " + "\n  ".join(sorted(set(parsers) - set(_baseline()["known"])))
        + "\n\nUse app.services.core.claude_tool_call.call_with_tool. Do not raise the "
          "baseline."
    )


def test_the_fence_stripper_count_never_rises():
    """Counted separately because it is the strongest signal: repairing a fenced reply
    is an admission that the free-form contract does not hold."""
    _, fences = sweep()
    recorded = _baseline()["fence_strippers"]
    assert len(fences) <= recorded, (
        f"markdown-fence repairs rose from {recorded} to {len(fences)}"
    )


def test_the_baseline_is_kept_honest():
    """A baseline nobody lowers is a baseline that has stopped meaning anything. When
    the real count falls below the recorded one, this asks for the number to come
    down — the same ratchet `check_ruff_gate` applies."""
    parsers, fences = sweep()
    recorded = _baseline()
    assert len(parsers) >= recorded["free_form_parsers"] - 3, (
        f"the count dropped to {len(parsers)} against a baseline of "
        f"{recorded['free_form_parsers']} — lower the baseline in the same change that "
        "fixed them, so the next regression is caught against the new floor"
    )
    assert len(fences) >= recorded["fence_strippers"] - 3, (
        f"fence-strippers dropped to {len(fences)} against {recorded['fence_strippers']}"
    )


def test_the_shared_forced_tool_helper_exists_and_fails_loudly():
    """The whole migration depends on there being one thing to migrate TO. The correct
    shape was already hand-copied into four files before this existed."""
    src = (APP / "services" / "core" / "claude_tool_call.py").read_text(encoding="utf-8")
    assert "class ToolCallNotReturned" in src, (
        "the typed error is gone — an absent tool block would collapse back into None, "
        "which is what made the free-form parsers unable to tell a broken reply from an "
        "empty one"
    )
    assert '"tool_choice": {"type": "tool"' in src, "the tool call is no longer FORCED"
    assert "tracked_claude_call_async" in src, (
        "the helper no longer goes through the tracked call, so its spend would be "
        "invisible to every cost view (pipeline convention 10)"
    )


def test_the_transport_exemption_is_still_transport():
    """The one exempted function must still BE the streaming transport for a forced
    tool call. An exemption nobody re-checks is how a real parser ends up living behind
    one — so this asserts the two facts the exemption rests on, and fails the moment
    either stops being true.
    """
    helper = (APP / "services" / "core" / "claude_helper.py").read_text(encoding="utf-8")
    body = ast.get_source_segment(
        helper,
        next(
            (n for n in ast.walk(ast.parse(helper))
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == "tracked_claude_stream_async"),
            None,
        ),
    )
    assert body, (
        "tracked_claude_stream_async is gone — drop it from _TRANSPORT rather than "
        "leaving an exemption that covers nothing"
    )
    assert "input_json_delta" in body, (
        "the exempted function no longer decodes tool-input deltas, so its json.loads "
        "is parsing something else — take it off _TRANSPORT and count it"
    )

    caller = (APP / "services" / "core" / "claude_tool_call.py").read_text(encoding="utf-8")
    stream = ast.get_source_segment(
        caller,
        next(
            (n for n in ast.walk(ast.parse(caller))
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name == "stream_with_tool"),
            None,
        ),
    )
    assert stream and '"tool_choice": {"type": "tool"' in stream, (
        "stream_with_tool no longer FORCES the tool, so the streamed reply is free-form "
        "after all and the transport exemption is unearned"
    )


def test_the_helper_never_returns_a_default():
    """`extract_tool_input` raising rather than returning {} is the point: a forced
    tool_choice that produced no block is a broken contract, not an empty answer."""
    src = (APP / "services" / "core" / "claude_tool_call.py").read_text(encoding="utf-8")
    body = ast.get_source_segment(
        src,
        next(n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.FunctionDef) and n.name == "extract_tool_input"),
    )
    assert "return {}" not in body and "return None" not in body, (
        "extract_tool_input grew a default return"
    )
