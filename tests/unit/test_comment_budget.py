"""A comment says what the code IS, not how it got that way.

The narrative belongs in the commit message or the issue, not in every context window that ever
loads the file. Structured sections (Args/Returns/Raises/...) never count, so a documented
signature is never what trips this.
"""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "site-packages", ".eggs",
}


def _load_budget():
    """Load the predicate by path — MIVAA CI installs pytest and no app dependencies."""
    path = ROOT / "scripts" / "comment_budget.py"
    spec = importlib.util.spec_from_file_location("comment_budget", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUDGET = _load_budget()


def _python_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in sorted(filenames):
            if name.endswith(".py"):
                yield Path(dirpath) / name


class TestThePredicate:
    def test_counts_narrative_lines_only(self):
        assert BUDGET.prose_lines("One.\nTwo.\n\nThree.") == 3

    def test_stops_counting_at_a_structured_section(self):
        body = "Summary.\n\nArgs:\n    a: one\n    b: two\n    c: three\n    d: four\n    e: five\n"
        assert BUDGET.prose_lines(body) == 1

    def test_a_documented_signature_is_never_over_budget(self):
        body = "Summary.\n\nArgs:\n" + "".join(f"    p{i}: value\n" for i in range(20))
        assert BUDGET.prose_lines(body) <= BUDGET.MAX_PROSE_LINES

    def test_a_directive_is_not_prose(self):
        assert BUDGET.is_directive("# noqa: E501")
        assert BUDGET.is_directive("# type: ignore")
        assert not BUDGET.is_directive("# Ordinary explanation.")

    def test_collapse_keeps_the_opening_paragraph_and_every_section(self):
        body = "What this is.\n\nWhy, at length.\nAnd more.\n\nArgs:\n    a: one\n"
        head, sections = BUDGET.collapse(body)
        assert head == ["What this is."]
        assert sections[0].strip() == "Args:"
        assert any("a: one" in line for line in sections)

    def test_flags_a_docstring_over_the_budget(self):
        body = "\\n".join(f"Line {i}." for i in range(BUDGET.MAX_PROSE_LINES + 3))
        source = f'def f():\n    """{body}"""\n    return 1\n'
        found = [o for o in BUDGET.find_offenders(source.replace("\\n", "\n    "))]
        assert found, "an over-budget docstring must be reported"

    def test_a_hash_run_counts_as_one_comment(self):
        source = "".join(f"# line {i}\n" for i in range(9)) + "x = 1\n"
        found = BUDGET.find_offenders(source)
        assert len(found) == 1
        assert found[0]["kind"] == "comment"
        assert found[0]["prose"] == 9

    def test_cuts_at_a_sentence_that_ends_not_at_a_colon(self):
        body = (
            "One. Two ends here.\nThree runs on and on\nand on and on\nand on and on\n"
            "and on and on\nand on and on\nand on and on\nand finally stops.\n"
        )
        head, _ = BUDGET.collapse(body)
        assert " ".join(head) == "One. Two ends here."

    def test_keeps_the_budget_when_nothing_ends_inside_the_window(self):
        body = "\n".join(["no terminator here"] * 9)
        head, _ = BUDGET.collapse(body)
        assert len(head) == BUDGET.MAX_PROSE_LINES

    def test_reads_the_hash_colon_marker_so_its_separator_still_ends_a_paragraph(self):
        source = "#: First para.\n#:\n#: Second para.\n" + "#: more.\n" * 6 + "x = 1\n"
        found = BUDGET.find_offenders(source)
        assert found and found[0]["marker"] == "#:"
        head, _ = BUDGET.collapse(found[0]["body"])
        assert head == ["First para."]

    def test_blank_comments_leaves_code_but_removes_prose(self):
        source = 'def f():\n    """A docstring naming get_workspace_context."""\n    # and a comment\n    return 1\n'
        blanked = BUDGET.blank_comments(source)
        assert "get_workspace_context" not in blanked
        assert "and a comment" not in blanked
        assert "return 1" in blanked
        assert len(blanked) == len(source), "offsets must survive, or every span shifts"

    def test_a_trailing_comment_is_not_part_of_a_run(self):
        source = "x = 1  # note\n" + "".join(f"# line {i}\n" for i in range(3)) + "y = 2\n"
        runs = BUDGET.comment_runs(source)
        assert [len(lines) for _, _, _, lines in runs] == [3]


class TestThePlatform:
    def test_no_comment_is_over_the_budget(self):
        files = list(_python_files())
        assert len(files) > 200, "the walk found almost nothing; check SKIP_DIRS"

        offenders = []
        for path in files:
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            for item in BUDGET.find_offenders(source):
                rel = path.relative_to(ROOT).as_posix()
                offenders.append(f"{rel}:{item['line']} ({item['prose']} prose lines)")

        assert not offenders, (
            f"{len(offenders)} comment(s) over {BUDGET.MAX_PROSE_LINES} prose lines. "
            "Run `python scripts/trim_comments.py`.\n" + "\n".join(offenders[:40])
        )

    def test_the_budget_is_declared_in_exactly_one_place(self):
        here = Path(__file__).resolve()
        declarers = []
        for path in _python_files():
            if path.name == "comment_budget.py" or path.resolve() == here:
                continue
            try:
                source = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "MAX_PROSE_LINES =" in source:
                declarers.append(path.relative_to(ROOT).as_posix())
        assert not declarers, f"a second copy of the budget has appeared: {declarers}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
