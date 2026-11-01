"""Documents whose ingestion is still running or ended less than SETTLING_HOURS ago.

An unattended repair sweep cannot tell "this image failed" from "the job has not reached
it yet", so it skips these documents; an explicit, targeted repair does not.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, List

SETTLING_HOURS = 24
ACTIVE_JOB_STATUSES = ("pending", "queued", "processing", "interrupted")


async def settling_document_ids(client: Any, hours: int = SETTLING_HOURS) -> List[str]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    query = (
        client.table("background_jobs")
        .select("document_id")
        .not_.is_("document_id", "null")
        .or_(f"status.in.({','.join(ACTIVE_JOB_STATUSES)}),updated_at.gte.{cutoff}")
    )
    rows = (await asyncio.to_thread(query.execute)).data or []
    return sorted({str(r["document_id"]) for r in rows if r.get("document_id")})


def outside_documents_clause(document_ids: List[str]) -> str:
    """A PostgREST `or(...)` member keeping rows of no listed document (NULL-safe)."""
    if not document_ids:
        return "document_id.is.null,document_id.not.is.null"
    return f"document_id.is.null,document_id.not.in.({','.join(document_ids)})"
