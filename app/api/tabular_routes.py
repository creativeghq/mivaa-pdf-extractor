"""POST /api/internal/tabular/ask — a question about a spreadsheet held in our storage.

Trusted-service only (x-cron-secret, the service-role JWT, the mk_ key): the caller is
inbox-api, which has already checked that the member may see the thread the file came
from and has RESERVED the member's credits (invariant 10). This route reads the object,
runs `TabularAgent`, and returns the answer with the tokens it spent so the caller can
settle the reservation. It does not meter on its own — two ledgers for one question is
the shape that made credit_transactions and ai_usage_logs disagree before.

`/api/internal` is excluded from the JWT middleware, so the gate on the route is the
gate (invariant 5).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.dependencies import require_trusted_service, resolve_workspace_id
from app.services.core.supabase_client import SupabaseClient, get_supabase_client
from app.services.tabular.loader import SUPPORTED_EXTENSIONS
from app.services.tabular.tabular_agent import DEFAULT_MODEL, TabularAgent

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/internal/tabular", tags=["Spreadsheet questions"])

MAX_FILE_BYTES = 15 * 1024 * 1024


class AskRequest(BaseModel):
    workspace_id: str
    user_id: Optional[str] = None
    storage_bucket: str
    storage_object_path: str
    file_name: str = Field(..., description="Used for the table name and the extension check")
    question: str
    extra_instructions: Optional[str] = Field(default=None, description="Operator glossary or conventions; appended to the SQL prompt")
    model: Optional[str] = None


@router.post("/ask", dependencies=[Depends(require_trusted_service)])
async def ask_spreadsheet(
    body: AskRequest,
    request: Request,
    supabase: SupabaseClient = Depends(get_supabase_client),
    claims: Optional[Dict[str, Any]] = Depends(require_trusted_service),
) -> Dict[str, Any]:
    # Bound the way every route binds a caller-supplied workspace (invariant 1): the
    # x-cron-secret path carries no claims and is trusted by construction; a token goes
    # through the shared rule. The thread check below then re-verifies the path against it.
    workspace_id = body.workspace_id if claims is None else await resolve_workspace_id(claims, body.workspace_id, request)
    if not workspace_id:
        raise HTTPException(status_code=400, detail="workspace_id is required")

    ext = body.file_name.rsplit(".", 1)[-1].lower() if "." in body.file_name else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"unsupported file type .{ext}; supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    # The object path must belong to the inbox prefix of a thread in THIS workspace: the caller
    # asserted the workspace, and the path is re-checked against the thread it names rather
    # than trusted as given (invariant 1 — a body-supplied path is not a boundary).
    if not body.storage_object_path.startswith("inbox/"):
        raise HTTPException(status_code=403, detail="only inbox attachments can be queried")
    thread_id = body.storage_object_path.split("/")[1] if body.storage_object_path.count("/") >= 2 else ""
    if not thread_id:
        raise HTTPException(status_code=403, detail="malformed attachment path")
    owner = (
        supabase.client.table("inbox_threads").select("id, workspace_id").eq("id", thread_id).limit(1).execute().data or []
    )
    if not owner or str(owner[0].get("workspace_id")) != str(workspace_id):
        raise HTTPException(status_code=404, detail="attachment not found")

    try:
        data = await asyncio.to_thread(supabase.client.storage.from_(body.storage_bucket).download, body.storage_object_path)
    except Exception as e:  # noqa: BLE001 — surfaced as a 502 with the storage reason
        raise HTTPException(status_code=502, detail=f"could not read the file from storage: {str(e)[:200]}") from e
    if not data:
        raise HTTPException(status_code=404, detail="attachment not found")
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"file is {len(data)} bytes; the limit is {MAX_FILE_BYTES}")

    agent = TabularAgent(workspace_id=str(workspace_id), user_id=body.user_id, model=body.model or DEFAULT_MODEL)
    try:
        return await agent.ask([(body.file_name, data)], body.question, extra_instructions=body.extra_instructions)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
