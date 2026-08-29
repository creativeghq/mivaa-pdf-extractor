"""No retired Claude model id reaches a call site.

A superseded model id is the quietest possible defect. It is a valid string, it imports, and the
provider either serves the old model — so you silently pay for and get worse output than you think
— or 404s a path nothing exercises in CI. Neither shows up as a failure anyone can see.

This one had spread. ``claude-opus-4-8`` was retired on the platform side, and 36 live call sites
HERE kept passing it: the vision classifier, segmentation, OCR, product creation and enrichment,
the embedding-to-text service and the whole RAG synthesis path. ``app/config.py`` had already moved
its DEFAULTS to ``claude-opus-5``. The defaults were right; the code was not reading them.

Current families: Claude 5 (``claude-opus-5``, ``claude-sonnet-5``, ``claude-fable-5``) and
``claude-haiku-4-5`` — Haiku has no 5 yet, so it is NOT retired.

Source-based on purpose: MIVAA's CI installs pytest and no app dependencies, so a test that
imports the services under check would not run at all. The platform repo carries the twin of this
file (``tests/unit/claudeModelGeneration.test.ts``); change one and change both.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = ["app", "modal_app", "scripts"]
SKIP_PARTS = {".venv", "venv", "__pycache__", "node_modules", ".git", "build", "dist"}

#: ``claude-haiku-4-5`` is deliberately absent — it is the current Haiku.
RETIRED = re.compile(
    r"claude-(?:opus-4-(?:6|7|8)|sonnet-4-(?:5|6|7)|3(?:-[a-z0-9-]+)?|2|instant)\b"
)

#: Files whose whole job is to name the old ids.
ALLOWED = {"tests/unit/test_claude_model_generation.py"}

_BLOCK_COMMENT = re.compile(r'("""|\'\'\')(?:.|\n)*?\1')
#: Anchored to the start of a line, matching the convention the other 26 sweeps in this suite
#: use. A bare ``#[^\n]*`` also matches a ``#`` INSIDE a string literal (``"#ff0000"``) and blanks
#: the rest of that line, which would hide a model id sitting after it.
_LINE_COMMENT = re.compile(r"^[ \t]*#.*$", re.MULTILINE)


def _strip_comments(src: str) -> str:
    """Drop docstrings and ``#`` comments before matching.

    A comment recording WHY a model was retired ("was claude-opus-4-8 at 3x the real price") is
    what makes the fix legible six months later. Deleting that history to satisfy a lint is the
    wrong trade, so the check only looks at code.
    """
    def _blank(match: re.Match[str]) -> str:
        # Blanked in place, not deleted: removing a comment shifts every line number after it, so
        # the failure message would point at the wrong line — worse than reporting none.
        return "\n" * match.group(0).count("\n")

    return _LINE_COMMENT.sub(_blank, _BLOCK_COMMENT.sub(_blank, src))


def _python_files() -> list[Path]:
    out: list[Path] = []
    for root in SCAN_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if SKIP_PARTS & set(path.parts):
                continue
            out.append(path)
    return out


def test_scans_a_non_empty_tree() -> None:
    """Guards against a walk that silently finds nothing and reports success."""
    assert len(_python_files()) > 100


def test_no_retired_claude_model_id_in_executable_code() -> None:
    hits: list[str] = []
    for path in _python_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOWED:
            continue
        code = _strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        for match in RETIRED.finditer(code):
            line = code.count("\n", 0, match.start()) + 1
            hits.append(f"{rel}:{line} -> {match.group(0)}")

    assert not hits, (
        "Retired Claude model ids in live code. Use claude-opus-5 / claude-sonnet-5 / "
        "claude-haiku-4-5 (Haiku 4.5 IS current):\n  " + "\n  ".join(hits)
    )
