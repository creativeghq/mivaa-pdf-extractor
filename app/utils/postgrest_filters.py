"""Escaping helpers for PostgREST filter values.

Separate contract from HTML escaping and from SQL quoting — do not reuse
`escape_like` for either. It exists for exactly one job: putting a
user-supplied term inside a `.like()` / `.ilike()` pattern without letting
that term act as a wildcard.
"""

# Postgres LIKE metacharacters, plus the backslash that escapes them.
# `*` is here because PostgREST accepts it as an alias for `%` in like/ilike
# (it rewrites the pattern before Postgres sees it), so an unescaped asterisk
# from a user is a wildcard even though Postgres itself would treat it as an
# ordinary character.
_LIKE_METACHARACTERS = ("\\", "%", "_", "*")


def escape_like(term: str) -> str:
    r"""Neutralise LIKE wildcards in a user-supplied search term.

    Returns the term with ``\``, ``%``, ``_`` and ``*`` backslash-escaped,
    ready to be wrapped in ``%...%`` by the caller. Postgres treats a
    backslash as the default LIKE escape character, so ``\%`` matches a
    literal percent sign.

    Caveat on ``*``: PostgREST rewrites ``*`` to ``%`` in like/ilike patterns,
    so an escaped ``\*`` reaches Postgres as ``\%`` and matches a literal
    ``%`` rather than a literal ``*``. Closing the wildcard is worth the
    mismatch — a bare ``*`` would otherwise match every row in the table.

    NOT safe for building ``or=(...)`` / ``in.(...)`` filter strings:
    PostgREST parses those as quoted values and consumes single backslashes,
    and commas and parentheses are structural there. This is for plain
    single-column ``.like()`` / ``.ilike()`` values only.
    """
    if not term:
        return ""
    escaped = term
    for char in _LIKE_METACHARACTERS:
        escaped = escaped.replace(char, "\\" + char)
    return escaped
