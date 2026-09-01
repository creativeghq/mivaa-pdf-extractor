"""Contract for calling a read-only PostgREST RPC over GET.

WHY THIS EXISTS
---------------
PostgREST maps `insert`, `upsert` AND `rpc` all onto POST. The central retry patch in
`app.services.core.supabase_client` therefore cannot tell a credit debit from a SELECT
wearing a function name, so it refuses to retry any of them. That rule is right — being
wrong toward "duplicate a write" costs a duplicate row, being wrong the other way costs
one surfaced error.

The cost of it landed on the crons. They call PostgREST once an hour, Supabase closes the
pooled keep-alive connection in between, and the first call after the gap raises
"Server disconnected". When that call is the `..._due` RPC that decides what to work on,
the whole tick is skipped — and pg_cron still records `succeeded`, because all it did was
queue the HTTP request. Ten of the twelve Sentry issues of this shape in the 17 days to
2026-09-01 were exactly that (MIVAA-5KH, -5KA, -5K7, -5JT, -5JS ...).

The fix is to stop lying about the method. Postgres already knows whether a function can
be repeated safely — that is `pg_proc.provolatile` — and PostgREST ENFORCES it: ask for
GET on a VOLATILE function and it answers 405. So there is no list of "safe" function
names here to drift out of date; the database keeps answering the question. GET then
lands in the retry patch's idempotent set and is retried transparently.

WHAT THIS MODULE GUARDS
-----------------------
GET carries its arguments in the query string, and two shapes do not survive that trip:

  * `None` — httpx renders it as an empty string, so a nullable `uuid` argument arrives
    as `''` and PostgREST fails the cast. Over POST the same value would have been a
    proper SQL NULL. That is a SILENT change of meaning, and it is why
    `read_document_chunk_span` (whose `p_product_id` is routinely None) is NOT called
    this way.
  * `list` / `tuple` / `set` / `dict` — a Postgres array argument has to arrive as the
    literal `{a,b}`; httpx instead repeats the key, which PostgREST does not read as an
    array. This is why `kb_read_doc_section` (`allowed_access_levels text[]`) and
    `expand_document_chunk_hits` (`p_chunk_ids uuid[]`) are NOT called this way.

Both would otherwise surface as a confusing server-side cast error a long way from the
call site, so they are refused HERE instead.

Stdlib only, on purpose: MIVAA's CI installs pytest and nothing else, so the guard test
can load this module by path and exercise the real function rather than a restatement of
it.
"""

from typing import Any, Mapping, Optional
from urllib.parse import urlencode

__all__ = ["read_rpc_param_error", "MAX_QUERY_STRING_LENGTH"]

#: Conservative ceiling for the generated query string. PostgREST sits behind Cloudflare,
#: which rejects long URLs with a 414 that says nothing about which argument was to blame.
MAX_QUERY_STRING_LENGTH = 4000


def read_rpc_param_error(func: str, params: Optional[Mapping[str, Any]]) -> Optional[str]:
    """Why `params` cannot be sent in a query string, or None if they can.

    Returns a reason naming the offending ARGUMENT, because the whole point is to fail
    where the mistake is rather than where PostgREST notices it.
    """
    if not params:
        return None

    for key, value in params.items():
        if value is None:
            return (
                f"argument {key!r} is None, which a query string cannot carry — httpx "
                f"sends it as an empty string and PostgREST fails the cast instead of "
                f"receiving SQL NULL. Call this RPC over POST."
            )
        if isinstance(value, (list, tuple, set, frozenset, dict)):
            return (
                f"argument {key!r} is a {type(value).__name__}; a Postgres array argument "
                f"must arrive as the literal '{{a,b}}' and httpx repeats the key instead. "
                f"Call this RPC over POST."
            )

    encoded = urlencode({str(k): str(v) for k, v in params.items()})
    if len(encoded) > MAX_QUERY_STRING_LENGTH:
        return (
            f"the arguments encode to {len(encoded)} characters, over the "
            f"{MAX_QUERY_STRING_LENGTH} ceiling for a query string. Call this RPC over POST."
        )

    return None
