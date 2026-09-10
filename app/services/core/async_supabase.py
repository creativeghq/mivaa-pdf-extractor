"""Async wrapper for the synchronous supabase-py client."""

import asyncio
from typing import Any


class AsyncQuery:
    """
    Wraps any supabase-py query builder object and makes `.execute()` awaitable.

    All fluent builder methods (.select, .eq, .insert, .update, .upsert, .delete,
    .in_, .is_, .order, .limit, .single, .neq, etc.) are proxied transparently.
    When one of those methods returns another query builder, the result is wrapped
    again so the chain stays async-aware all the way to `.execute()`.
    """

    __slots__ = ('_q',)

    def __init__(self, sync_query: Any) -> None:
        object.__setattr__(self, '_q', sync_query)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(object.__getattribute__(self, '_q'), name)
        if not callable(attr):
            return attr

        def _proxy(*args: Any, **kwargs: Any) -> Any:
            result = attr(*args, **kwargs)
            # Re-wrap if the result looks like a query builder (has .execute)
            if result is not None and hasattr(result, 'execute') and callable(result.execute):
                return AsyncQuery(result)
            return result

        return _proxy

    async def execute(self) -> Any:
        """Dispatch .execute() to the thread pool — never blocks the event loop."""
        return await asyncio.to_thread(object.__getattribute__(self, '_q').execute)


class AsyncSupabaseClient:
    """
    Drop-in async façade over the synchronous supabase-py Client.

    Wraps `.table()`, `.rpc()`, and `.schema()` so every chain ending in
    `.execute()` becomes awaitable. `.storage` and `.auth` are passed through
    unchanged (they handle their own async internally).
    """

    __slots__ = ('_client',)

    def __init__(self, sync_client: Any) -> None:
        object.__setattr__(self, '_client', sync_client)

    def table(self, name: str) -> AsyncQuery:
        return AsyncQuery(object.__getattribute__(self, '_client').table(name))

    def from_(self, name: str) -> AsyncQuery:
        return AsyncQuery(object.__getattribute__(self, '_client').from_(name))

    def rpc(self, fn: str, params: dict, get: bool = False) -> AsyncQuery:
        # `get` rides through because a read-only (STABLE/IMMUTABLE) RPC is called over
        # GET so the central retry patch may repeat it — see `read_rpc` in
        # supabase_client. A façade that quietly drops the keyword is not drop-in: the
        # call raises TypeError inside an enclosing `except Exception` and the work
        # simply never happens.
        return AsyncQuery(
            object.__getattribute__(self, '_client').rpc(fn, params, get=get)
        )

    def schema(self, schema_name: str) -> 'AsyncSupabaseClient':
        """Return a new AsyncSupabaseClient scoped to a different Postgres schema."""
        return AsyncSupabaseClient(
            object.__getattribute__(self, '_client').schema(schema_name)
        )

    @property
    def storage(self) -> Any:
        return object.__getattribute__(self, '_client').storage

    @property
    def auth(self) -> Any:
        return object.__getattribute__(self, '_client').auth
