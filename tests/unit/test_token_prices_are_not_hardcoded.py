"""A per-token USD rate written into Python is a wrong number that nothing raises on.

`ai_model_pricing` is the single USD source for this platform. Before this guard,
`image_processing_service` priced its own spend from a constant:

    # Claude Opus 4.7 pricing as of 2026-05-01: $15/M input, $75/M output.
    cost = (input_tokens / 1_000_000) * 15.0 + (output_tokens / 1_000_000) * 75.0

The comment names Opus 4.7. The call three lines above it asks for `claude-opus-4-8`.
So every image in every catalogue — a per-image hot path — was booked at a rate that
belonged to a different model, and had been since the model string was bumped without
the constant beneath it.

WHY THIS SHAPE SURVIVES
-----------------------
It is the silent-zero defect wearing a different hat. There is no failure: the
multiplication succeeds, a float lands in `ai_usage_logs`, every dashboard renders, and
the number is simply wrong. A typecheck cannot see it (a wrong price is a valid float),
an integrity probe cannot see it (the stored data is internally consistent), and the
only way to notice is to reconcile against the provider invoice — which nobody does per
image.

WHAT TO DO INSTEAD
------------------
`tracked_claude_call_async` / `call_with_tool` resolve the rate from `ai_model_pricing`
at log time, so a price change is a DB edit rather than a deploy, and a model bump
cannot leave its rate behind.

WATCHED TO FAIL: run against the pre-fix tree, `test_no_module_prices_tokens_itself`
reports image_processing_service.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"

SKIP_DIRS = {"__pycache__", ".venv", "venv", "node_modules"}

#: The module that legitimately holds a fallback rate table, and the logger that
#: resolves against `ai_model_pricing`. A rate here is the source, not a copy.
ALLOWED = {
    "app/services/core/ai_call_logger.py",
    "app/services/core/model_pricing.py",
}

#: `tokens / 1_000_000 * <rate>` and `tokens / 1000 * <rate>` — the two shapes a
#: hand-rolled per-token price takes. Matched on the DIVISOR plus a multiplication by a
#: float literal, because that combination does not occur for anything else.
_PER_TOKEN_MATH = re.compile(
    r"/\s*1[_,]?000[_,]?000\s*\)?\s*\*\s*\d+\.\d+"
    r"|/\s*1[_,]?000\s*\)?\s*\*\s*\d+\.\d+"
)


def _strip_comments(src: str) -> str:
    """The offending line is now QUOTED in a comment in the file that used to carry it,
    to record what was wrong. Reading prose as code would make this guard fire on its
    own documentation — the exact false positive that makes a finding unactionable."""
    src = re.sub(r'"""[\s\S]*?"""', "", src)
    src = re.sub(r"'''[\s\S]*?'''", "", src)
    src = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    return src


def sweep():
    """Relative paths that compute a USD cost from a literal per-token rate."""
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
        if _PER_TOKEN_MATH.search(src):
            offenders.append(rel)
    return sorted(offenders)


def test_no_module_prices_tokens_itself():
    offenders = sweep()
    assert not offenders, (
        "these modules compute a USD cost from a hardcoded per-token rate:\n  "
        + "\n  ".join(offenders)
        + "\n\n`ai_model_pricing` is the one USD source. Call through "
          "`tracked_claude_call_async` / `call_with_tool` and let the logger resolve "
          "the rate — a constant does not fail, it just books the wrong number "
          "forever, and a model bump leaves it behind silently."
    )


def test_the_documented_offender_is_still_quoted_where_it_happened():
    """The removed line is kept as a comment in the file it was removed from. If someone
    deletes that note the guard loses the only record of what it is defending against —
    and this test is how anyone reading the sweep finds the story."""
    src = (APP / "services" / "images" / "image_processing_service.py").read_text(encoding="utf-8")
    assert "Opus 4.7 pricing" in src, (
        "the note explaining why image classification stopped pricing itself is gone; "
        "keep it, or move it into this docstring so the reason survives"
    )


def test_the_sweep_can_actually_see_the_shape_it_hunts():
    """A regex guard that matches nothing is indistinguishable from a clean tree. This
    pins the matcher against the exact line that was removed, plus the /1000 variant."""
    assert _PER_TOKEN_MATH.search(
        "cost = (input_tokens / 1_000_000) * 15.0 + (output_tokens / 1_000_000) * 75.0"
    )
    assert _PER_TOKEN_MATH.search("cost = (tok / 1000) * 0.25")
    assert not _PER_TOKEN_MATH.search("ratio = total / 1_000_000"), (
        "the matcher fires on a bare division — it needs the multiplication by a rate, "
        "or it will report every unit conversion in the tree"
    )
