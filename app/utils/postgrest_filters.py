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
    r"""Neutralise LIKE wildcards in a user-supplied search term."""
    if not term:
        return ""
    escaped = term
    for char in _LIKE_METACHARACTERS:
        escaped = escaped.replace(char, "\\" + char)
    return escaped
