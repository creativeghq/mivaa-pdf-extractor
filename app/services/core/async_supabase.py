"""
Async wrapper for the synchronous supabase-py client.

supabase-py 2.x ships only a synchronous client. Every `.execute()` call
makes a blocking HTTP request that holds the asyncio event loop hostage for
50-200 ms per call. At batch scale (10 products × 15 DB calls each = 150
calls per batch) this freezes all other FastAPI requests for up to 30 s.

This module intercepts the query builder chain at the final `.execute()` step
and offloads it to `asyncio.to_thread()`, which runs it in the ThreadPoolExecutor
configured in main.py's lifespan (max_workers=20, thread_name_prefix="supabase-io").

Usage — drop-in replacement, just add `await` and use `self.db` instead of
`self.supabase`:

    # Before (blocks event loop):
    result = self.supabase.table('products').insert(record).execute()

    # After (non-blocking):
    result = await self.db.table('products').insert(record).execute()

    # RPC:
    result = await self.db.rpc('merge_background_job_metadata', {...}).execute()

    # Schema-qualified (vecs):
    result = await self.db.schema('vecs').from_('image_slig_embeddings').select('*').execute()
"""

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

    def rpc(self, fn: str, params: dict) -> AsyncQuery:
        return AsyncQuery(object.__getattribute__(self, '_client').rpc(fn, params))

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
