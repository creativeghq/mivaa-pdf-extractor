"""Documents whose ingestion is still running or ended within the settling window (24h).

An unattended repair sweep cannot tell "this image failed" from "the job has not reached
it yet", so it skips these documents; an explicit, targeted repair does not. The rule and
the window are the SQL function `ingestion_settling_document_ids`, which the edge agent
calls too.
"""
import asyncio
from typing import Any, List


async def settling_document_ids(client: Any) -> List[str]:
    call = client.rpc("ingestion_settling_document_ids", {})
    rows = (await asyncio.to_thread(call.execute)).data or []
    return sorted({str(r) for r in rows if r})


def outside_documents_clause(document_ids: List[str]) -> str:
    """A PostgREST `or(...)` member keeping rows of no listed document (NULL-safe)."""
    if not document_ids:
        return "document_id.is.null,document_id.not.is.null"
    return f"document_id.is.null,document_id.not.in.({','.join(document_ids)})"
