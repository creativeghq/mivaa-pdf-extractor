"""One switch for a job that builds no vectors until its document is approved (#513).

The orchestrator sets it once from the job's `defer_embedding`; asyncio tasks and `to_thread` copy
the context, so every embedding site in that run sees it without a parameter threaded through.
Work the job hands to another request (factory enrichment over HTTP) cannot see a context, so the
document also carries `metadata.embedding_deferred` until `embed_document` clears it.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Any, List, Optional

_DEFERRED: ContextVar[bool] = ContextVar("embedding_deferred", default=False)


class EmbeddingDeferred(Exception):
    """Raised inside an embedding block to leave it without counting a failure."""


def defer_embeddings(on: bool) -> Token:
    return _DEFERRED.set(bool(on))


def embeddings_deferred() -> bool:
    return _DEFERRED.get()


def document_deferred(db: Any, document_id: Optional[str]) -> bool:
    """The document's own marker, for work running outside the job's context."""
    if not document_id:
        return False
    db = getattr(db, "client", db)
    rows = db.table("documents").select("metadata").eq("id", document_id).limit(1).execute().data or []
    return bool(rows and (rows[0].get("metadata") or {}).get("embedding_deferred"))


def deferred_document_ids(db: Any) -> List[str]:
    """Documents waiting on their embed run: no backfill may reach them first."""
    db = getattr(db, "client", db)
    rows = db.table("documents").select("id").eq("metadata->>embedding_deferred", "true").execute().data or []
    return [r["id"] for r in rows]


def outside_documents(column: str, ids: List[str]) -> str:
    """A PostgREST `or` body keeping rows with no document or a document not in `ids`."""
    return f"{column}.is.null,{column}.not.in.({','.join(ids)})"


def mark_document(db: Any, document_id: str, deferred: bool) -> None:
    """Set or clear the document's marker (read-modify-write of documents.metadata)."""
    db = getattr(db, "client", db)
    rows = db.table("documents").select("metadata").eq("id", document_id).limit(1).execute().data or []
    meta = dict((rows[0].get("metadata") if rows else None) or {})
    if deferred:
        meta["embedding_deferred"] = True
    else:
        meta.pop("embedding_deferred", None)
    db.table("documents").update({"metadata": meta}).eq("id", document_id).execute()
