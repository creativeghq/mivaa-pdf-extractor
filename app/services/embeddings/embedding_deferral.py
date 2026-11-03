"""One switch for a job that builds no vectors until its document is approved (#513).

The orchestrator sets it once from the job's `defer_embedding`; asyncio tasks and `to_thread` copy
the context, so every embedding site in that run sees it without a parameter threaded through.
The vectors come later from `embed_document`, in one pass.
"""

from __future__ import annotations

from contextvars import ContextVar, Token

_DEFERRED: ContextVar[bool] = ContextVar("embedding_deferred", default=False)


class EmbeddingDeferred(Exception):
    """Raised inside an embedding block to leave it without counting a failure."""


def defer_embeddings(on: bool) -> Token:
    return _DEFERRED.set(bool(on))


def embeddings_deferred() -> bool:
    return _DEFERRED.get()
