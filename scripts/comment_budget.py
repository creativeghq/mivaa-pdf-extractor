"""The comment budget for the Python side: how long a comment may be, in ONE place.

Read by the codemod and by the guard test. Stdlib only, and loadable by path, because MIVAA CI
installs pytest and nothing else.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize

MAX_PROSE_LINES = 6

#: Structured docstring sections. Everything from the first one on is kept and never counted,
#: so a documented signature is never what trips the budget.
SECTION_RE = re.compile(
    r"^(Args|Arguments|Parameters|Params|Returns|Return|Yields|Yield|Raises|Except|Attributes|"
    r"Members|Example|Examples|Note|Notes|Warning|Warnings|Warns|See Also|References|Todo|"
    r"Usage|Methods|Other Parameters)\s*:?\s*$",
    re.IGNORECASE,
)

#: A comment that instructs a tool is not prose. Never counted, never trimmed.
DIRECTIVE_RE = re.compile(
    r"#\s*(noqa|type:|pragma:|pylint:|mypy:|flake8:|fmt:\s*(on|off)|ruff:|isort:|nosec|coding[:=]|!)"
)


def prose_lines(body: str) -> int:
    """Count the narrative lines in a docstring or comment block."""
    count = 0
    for line in body.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if SECTION_RE.match(stripped) or set(stripped) <= {"-", "=", "~"}:
            break
        count += 1
    return count


def split_sections(body: str) -> tuple[list[str], list[str]]:
    """Split a docstring into its narrative lines and everything from the first section on."""
    lines = body.split("\n")
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped and (SECTION_RE.match(stripped) or set(stripped) <= {"-", "=", "~"}):
            return lines[:index], lines[index:]
    return lines, []


def collapse(body: str, max_lines: int = MAX_PROSE_LINES) -> tuple[list[str], list[str]]:
    """The docstring cut to budget: its opening paragraph, plus every structured section."""
    narrative, sections = split_sections(body)

    head: list[str] = []
    for line in narrative:
        if not line.strip():
            if head:
                break
            continue
        head.append(line.strip())
    if len(head) > max_lines:
        window = head[:max_lines]
        cut = -1
        for index in range(len(window) - 1, -1, -1):
            if window[index].rstrip().endswith((".", "!", "?", ":", ";", ")")):
                cut = index
                break
        head = window[: cut + 1] if cut >= 0 else window

    while sections and not sections[0].strip():
        sections.pop(0)
    while sections and not sections[-1].strip():
        sections.pop()
    return head, sections


def is_directive(text: str) -> bool:
    return bool(DIRECTIVE_RE.search(text))


def docstring_nodes(tree: ast.AST):
    """Every node that owns a docstring, paired with the string expression holding it."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            yield node, first.value


def comment_runs(source: str):
    """Consecutive own-line `#` comments, as one run each: (start_row, end_row, indent, lines)."""
    runs: list[tuple[int, int, str, list[str]]] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return runs

    lines = source.split("\n")
    current: list[tuple[int, str]] = []

    def flush() -> None:
        if not current:
            return
        first_row = current[0][0]
        indent = lines[first_row - 1][: len(lines[first_row - 1]) - len(lines[first_row - 1].lstrip())]
        runs.append((first_row, current[-1][0], indent, [text for _, text in current]))
        current.clear()

    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        row = token.start[0]
        if lines[row - 1][: token.start[1]].strip():
            flush()  # a trailing comment; never part of a run
            continue
        if current and row != current[-1][0] + 1:
            flush()
        current.append((row, token.string))
    flush()
    return runs


def find_offenders(source: str) -> list[dict]:
    """Every docstring and comment run in `source` that is over budget."""
    out: list[dict] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out

    for _, const in docstring_nodes(tree):
        text = const.value
        if is_directive(text):
            continue
        count = prose_lines(text)
        if count > MAX_PROSE_LINES:
            out.append({"kind": "docstring", "line": const.lineno, "prose": count, "node": const})

    for start, end, indent, lines in comment_runs(source):
        body = "\n".join(re.sub(r"^#+ ?", "", line.strip()) for line in lines)
        if is_directive("\n".join(lines)):
            continue
        count = prose_lines(body)
        if count > MAX_PROSE_LINES:
            out.append(
                {"kind": "comment", "line": start, "end": end, "indent": indent,
                 "prose": count, "lines": lines, "body": body}
            )
    return out


def blank_comments(source: str) -> str:
    """`source` with comments and docstrings replaced by spaces, same length, same line numbers.

    A guard test that greps source for a marker must read CODE. Searching the raw text lets a
    comment that merely NAMES a gate satisfy the check for a route that has none.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    starts = [0]
    for index, char in enumerate(source):
        if char == "\n":
            starts.append(index + 1)

    spans: list[tuple[int, int]] = []
    for _, const in docstring_nodes(tree):
        if const.end_lineno is None:
            continue
        spans.append(
            (starts[const.lineno - 1] + const.col_offset,
             starts[const.end_lineno - 1] + const.end_col_offset)
        )
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                begin = starts[token.start[0] - 1] + token.start[1]
                spans.append((begin, begin + len(token.string)))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass

    chars = list(source)
    for begin, end in spans:
        for index in range(begin, min(end, len(chars))):
            if chars[index] != "\n":
                chars[index] = " "
    return "".join(chars)


def code_signature(source: str) -> str:
    """The module's code with every docstring removed. Two of these must match, or code changed."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            node.body.pop(0)
            if not node.body:
                node.body.append(ast.Pass())
    return ast.dump(tree)
