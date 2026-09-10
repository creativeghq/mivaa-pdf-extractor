#!/usr/bin/env python
"""Cut every over-budget docstring and comment run to its opening paragraph.

Usage: python scripts/trim_comments.py [--dry-run] [--list] [path ...]

Structured sections (Args/Returns/Raises/...) are kept in full; only the narrative is cut. A file
is written only when its code, compared as an AST with docstrings removed, is unchanged.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from comment_budget import (  # noqa: E402
    MAX_PROSE_LINES,
    code_signature,
    collapse,
    find_offenders,
)

SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", "site-packages",
}
PREFIXES = ("r", "b", "u", "f", "rb", "br", "fr", "rf", "R", "B", "U", "F")


def walk(root: str) -> list[str]:
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(".py"):
                found.append(os.path.join(dirpath, name))
    return sorted(found)


def line_starts(source: str) -> list[int]:
    starts = [0]
    for index, char in enumerate(source):
        if char == "\n":
            starts.append(index + 1)
    return starts


def quote_of(source: str, start: int) -> tuple[str, str]:
    """The docstring's prefix and quote characters, so the rewrite keeps its shape."""
    head = source[start:start + 8]
    prefix = ""
    for candidate in sorted(PREFIXES, key=len, reverse=True):
        if head.startswith(candidate) and head[len(candidate):len(candidate) + 1] in "\"'":
            prefix = candidate
            head = head[len(candidate):]
            break
    for quote in ('"""', "'''", '"', "'"):
        if head.startswith(quote):
            return prefix, quote
    return prefix, '"""'


def rebuild_docstring(head: list[str], sections: list[str], indent: str, prefix: str, quote: str) -> str:
    if not head and not sections:
        return f"{prefix}{quote}{quote}"
    if not sections and len(head) <= 1:
        body = head[0] if head else ""
        return f"{prefix}{quote}{body}{quote}"
    parts = [head[0] if head else ""]
    parts.extend(indent + line for line in head[1:])
    if sections:
        parts.append("")
        parts.extend(sections)
    parts.append(indent)
    return f"{prefix}{quote}" + "\n".join(parts) + quote


def process(source: str) -> tuple[str, int, int]:
    offenders = find_offenders(source)
    if not offenders:
        return source, 0, 0

    starts = line_starts(source)
    edits = []

    for item in offenders:
        if item["kind"] == "docstring":
            node = item["node"]
            begin = starts[node.lineno - 1] + node.col_offset
            end = starts[node.end_lineno - 1] + node.end_col_offset
            prefix, quote = quote_of(source, begin)
            if quote not in ('"""', "'''"):
                continue
            head, sections = collapse(node.value, MAX_PROSE_LINES)
            joined = "\n".join(head + sections)
            single_line = not sections and len(head) <= 1
            # A backslash re-escapes differently once the decoded value is written back, and the
            # quote run would close the string early. Only the one-line form can be ended by a
            # trailing quote; the multi-line form closes on its own indent.
            if quote in joined or "\\" in joined:
                continue
            if single_line and joined.endswith(('"', "'")):
                continue
            indent = " " * node.col_offset
            replacement = rebuild_docstring(head, sections, indent, prefix, quote)
            before = end - begin
            edits.append((begin, end, replacement, source[begin:end].count("\n") - replacement.count("\n")))
            del before
        else:
            begin = starts[item["line"] - 1]
            end_row = item["end"]
            end = starts[end_row - 1] + len(source.split("\n")[end_row - 1])
            head, sections = collapse(item["body"], MAX_PROSE_LINES)
            kept = head + sections
            indent = item["indent"]
            marker = item.get("marker", "#")
            replacement = "\n".join(
                (indent + marker + " " + line) if line.strip() else (indent + marker) for line in kept
            )
            if not kept:
                replacement = ""
            edits.append((begin, end, replacement, len(item["lines"]) - len(kept)))

    out = source
    for begin, end, replacement, _ in sorted(edits, key=lambda e: -e[0]):
        out = out[:begin] + replacement + out[end:]
    return out, len(edits), sum(edit[3] for edit in edits)


def main() -> int:
    argv = sys.argv[1:]
    dry = "--dry-run" in argv
    listing = "--list" in argv
    roots = [a for a in argv if not a.startswith("--")] or ["app", "scripts", "tests", "modal_app"]

    targets: list[str] = []
    for root in roots:
        if os.path.isdir(root):
            targets.extend(walk(root))
        elif os.path.isfile(root):
            targets.append(root)

    changed = trimmed = removed = refused = 0
    for path in targets:
        try:
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
        except (OSError, UnicodeDecodeError):
            continue

        if listing:
            for item in find_offenders(source):
                print(f"{item['prose']}\t{path}:{item['line']}")
                trimmed += 1
            continue

        try:
            out, count, lines = process(source)
        except (SyntaxError, IndexError):
            continue
        if out == source or count == 0:
            continue

        try:
            if code_signature(out) != code_signature(source):
                refused += 1
                print(f"REFUSED (code would change): {path}")
                continue
        except SyntaxError:
            refused += 1
            print(f"REFUSED (would not parse): {path}")
            continue

        if not dry:
            with open(path, "w", encoding="utf-8", newline="") as handle:
                handle.write(out)
        changed += 1
        trimmed += count
        removed += lines

    if listing:
        print(f"\n{trimmed} comments over {MAX_PROSE_LINES} prose lines.")
    else:
        prefix = "[dry-run] " if dry else ""
        print(f"{prefix}files {changed}, comments trimmed {trimmed}, lines removed {removed}, refused {refused}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
