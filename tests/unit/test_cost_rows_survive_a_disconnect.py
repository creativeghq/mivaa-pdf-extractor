"""
Guard: every `ai_usage_logs` write is REPEAT-SAFE, so a transient blip can neither
drop the cost row nor book it twice.

WHY THIS EXISTS
---------------
`_install_postgrest_retry_once` retries transient PostgREST failures, but it refuses to
repeat a bare POST — after a disconnect it cannot tell whether the server committed, and
repeating an INSERT would write the row twice. That refusal is correct. Its price is that
the row is DROPPED instead, and every `ai_usage_logs` writer swallows its own exception,
so the spend simply stops being recorded.

Both halves of that are the same silent-zero shape and neither raises:

  - dropped  -> the cost view under-reports. Sentry MIVAA-5KP (mention-monitoring),
                MIVAA-5KN (seo-toolkit) and MIVAA-5KE (embeddings) were all one lost row.
  - doubled  -> the cost view over-reports. `ai_call_logger` buffered failed rows in a
                dead-letter deque and re-INSERTed them with no key, which is exactly the
                duplicate the retry patch declines to risk, taken by hand one layer above it.

`repeatable_insert` closes both: it mints the primary key client-side, so a replay of a
write that DID land collapses onto it, and `.upsert()` carries the
`Prefer: resolution=merge-duplicates` header the patch already whitelists — so the retry
covers the write. A row that can be replayed safely does not need to be dropped.

The rule is structural rather than a list of known call sites, because the six writers
that existed when this was written are not the ones that will break it.

Source-based: imports no app module and touches no DB, so it runs in MIVAA's
pytest-only CI (no application dependencies are installed there).
"""

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[1].parent
_APP = _ROOT / "app"

#: Tables whose rows are money. A lost or doubled row here is a wrong number that
#: every downstream cost view reports as fact.
#:
#: `ai_call_logs` belongs here even though the cost MIRROR lives in `ai_usage_logs`:
#: `log_ai_call` writes the mirror only `if result.data`, so a dropped call row takes
#: the cost row with it. Guarding one table and not the other leaves the busiest path
#: in the platform exactly as exposed as it was.
_LEDGER_TABLES = ("ai_usage_logs", "ai_call_logs")

#: `sb.table("ai_usage_logs").insert(` — the shape that cannot be retried. Either quote
#: style: the Firecrawl writer used single quotes and a grep for the double-quoted form
#: reported this file clean.
_PLAIN_INSERT = re.compile(
    r"""table\(\s*["'](?P<table>%s)["']\s*\)\s*\.\s*insert\(""" % "|".join(_LEDGER_TABLES)
)


def _code_only(text: str) -> str:
    """The source with PROSE blanked — comments and docstrings — and nothing else.

    `credits_integration_service` DOCUMENTS this anti-pattern in its module docstring —
    it is the file that fixed it — so a raw text scan accuses the one writer that got it
    right. Blanking rather than deleting keeps every line number honest.

    Only docstrings, never string literals in general: the table name this scans for IS a
    string literal, so blanking all of them makes the guard match nothing and pass
    vacuously on a codebase full of offenders. The first version of this did exactly that.
    `test_the_guard_sees_a_planted_offender` below is what holds the line.
    """
    lines = text.splitlines(keepends=True)

    def blank(r1, c1, r2, c2):
        for row in range(r1, r2 + 1):
            if row - 1 >= len(lines):
                return
            line = lines[row - 1]
            start = c1 if row == r1 else 0
            end = c2 if row == r2 else len(line.rstrip("\r\n"))
            lines[row - 1] = line[:start] + " " * max(0, end - start) + line[end:]

    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                blank(tok.start[0], tok.start[1], tok.end[0], tok.end[1])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text  # Unparseable: scan it raw rather than skipping it silently.

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "".join(lines)

    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            and first.end_lineno is not None
        ):
            blank(first.lineno, first.col_offset, first.end_lineno, first.end_col_offset)

    return "".join(lines)


def _python_sources():
    for path in sorted(_APP.rglob("*.py")):
        yield path, _code_only(path.read_text(encoding="utf-8"))


def test_no_ledger_table_is_written_with_a_plain_insert():
    """A cost row goes through `repeatable_insert`, never `.insert()`."""
    offenders = []
    for path, text in _python_sources():
        for match in _PLAIN_INSERT.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(_ROOT)}:{line} -> {match.group('table')}")

    assert not offenders, (
        "These write a cost row with a bare `.insert()`, which the PostgREST retry patch "
        "refuses to repeat — so a transient disconnect drops the row and the spend goes "
        "unrecorded.\n\nUse `repeatable_insert(client, table, row)` from "
        "app.services.core.supabase_client instead.\n\n  " + "\n  ".join(offenders)
    )


def test_the_guard_sees_a_planted_offender():
    """The scan above must actually match. A guard that cannot fire passes forever.

    `_code_only` blanks prose so a file that merely DESCRIBES the anti-pattern is not
    accused. Blank one category too many — string literals — and the table name the regex
    keys on disappears with it, so every real writer reads as clean. That is not
    hypothetical: it is what the first version of this file did, and the headline test
    went green with two genuine offenders in the tree.
    """
    planted = (
        'def writer(sb):\n'
        '    sb.table("ai_usage_logs").insert({"operation_type": "x"}).execute()\n'
    )
    assert _PLAIN_INSERT.search(_code_only(planted)), (
        "_code_only destroyed the pattern the scan matches on, so "
        "test_no_ledger_table_is_written_with_a_plain_insert can never fail."
    )

    described = '"""We used to call table("ai_usage_logs").insert(...) here."""\n'
    assert not _PLAIN_INSERT.search(_code_only(described)), (
        "_code_only must still blank docstrings, or a file that documents the "
        "anti-pattern is reported as committing it."
    )


def test_repeatable_insert_mints_a_key_and_upserts():
    """The helper's two halves are what make the retry safe; neither is optional.

    Without the minted id an upsert has nothing to collapse onto and still duplicates;
    without the upsert the request is a bare POST the patch will not repeat.
    """
    source = _code_only(
        (_APP / "services" / "core" / "supabase_client.py").read_text(encoding="utf-8")
    )
    body = source.split("def repeatable_insert(", 1)
    assert len(body) == 2, "repeatable_insert has been removed or renamed"
    body = body[1].split("\ndef ", 1)[0]

    assert "setdefault(id_field" in body, (
        "repeatable_insert must mint the primary key client-side — that is the whole "
        "reason a replay is safe."
    )
    assert ".upsert(" in body, (
        "repeatable_insert must use .upsert(): the `Prefer: resolution=merge-duplicates` "
        "header is what `_is_safe_to_repeat` whitelists. A plain .insert() is not retried."
    )


def test_the_retry_patch_still_whitelists_upserts():
    """`repeatable_insert` is only useful while the patch honours `resolution=`.

    If `_is_safe_to_repeat` stops reading the Prefer header, every cost row silently goes
    back to being dropped — the helper would still look correct at all six call sites.
    """
    source = _code_only(
        (_APP / "services" / "core" / "supabase_client.py").read_text(encoding="utf-8")
    )
    guard = source.split("def _is_safe_to_repeat(", 1)
    assert len(guard) == 2, "_is_safe_to_repeat has been removed or renamed"
    guard = guard[1].split("\n    try:", 1)[0]

    assert "resolution=" in guard, (
        "_is_safe_to_repeat no longer recognises an upsert by its Prefer header, so "
        "repeatable_insert's writes are no longer retried."
    )


def test_the_dead_letter_row_carries_its_key_before_the_first_attempt():
    """`ai_call_logger` buffers a failed row and replays it, so the id must predate the try.

    Minting inside the helper is enough for a single attempt. It is NOT enough here: each
    replay would take a fresh id and re-book spend that may already be recorded. The id has
    to live on the buffered dict itself.
    """
    source = _code_only(
        (_APP / "services" / "core" / "ai_call_logger.py").read_text(encoding="utf-8")
    )
    entry = source.split("usage_entry = {", 1)
    assert len(entry) == 2, "usage_entry has been removed or renamed"
    head = entry[1].split("}", 1)[0]

    assert '"id"' in head, (
        "usage_entry must carry its own `id`. Without it, a row replayed from "
        "_dead_letter_usage_rows gets a new key on every attempt and duplicates the spend "
        "of a write that already landed."
    )
