"""POST /api/internal/drawings/* — geometry read from a project drawing PDF held in our storage."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import require_trusted_service
from app.services.core.supabase_client import SupabaseClient, get_supabase_client
from app.services.drawings import analyze_pdf, find_similar_in_pdf

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/internal/drawings", tags=["Drawing geometry"])

ALLOWED_BUCKET = "pdf-documents"
ALLOWED_PREFIX = "project-docs/"
MAX_FILE_BYTES = 60 * 1024 * 1024


class AnalyzeRequest(BaseModel):
    storage_bucket: str
    storage_object_path: str
    max_pages: int = Field(default=60, ge=1, le=500)
    units_per_page_width: Optional[float] = Field(default=None, gt=0)


class FindSimilarRequest(BaseModel):
    storage_bucket: str
    storage_object_path: str
    page: int = Field(..., ge=1)
    bbox: List[float] = Field(..., min_length=4, max_length=4)


async def _download(supabase: SupabaseClient, bucket: str, path: str) -> bytes:
    if bucket != ALLOWED_BUCKET or not path.startswith(ALLOWED_PREFIX) or ".." in path:
        raise HTTPException(status_code=403, detail="only project drawings can be analysed")
    try:
        data = await asyncio.to_thread(supabase.client.storage.from_(bucket).download, path)
    except Exception as e:  # noqa: BLE001 — surfaced as a 502 with the storage reason
        raise HTTPException(status_code=502, detail=f"could not read the file from storage: {str(e)[:200]}") from e
    if not data:
        raise HTTPException(status_code=404, detail="drawing not found")
    if len(data) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"file is {len(data)} bytes; the limit is {MAX_FILE_BYTES}")
    return data


@router.post("/analyze", dependencies=[Depends(require_trusted_service)])
async def analyze_drawing(
    body: AnalyzeRequest,
    supabase: SupabaseClient = Depends(get_supabase_client),
) -> Dict[str, Any]:
    data = await _download(supabase, body.storage_bucket, body.storage_object_path)
    try:
        return await asyncio.to_thread(analyze_pdf, data, body.max_pages, body.units_per_page_width)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/find-similar", dependencies=[Depends(require_trusted_service)])
async def find_similar_symbols(
    body: FindSimilarRequest,
    supabase: SupabaseClient = Depends(get_supabase_client),
) -> Dict[str, Any]:
    data = await _download(supabase, body.storage_bucket, body.storage_object_path)
    try:
        return await asyncio.to_thread(find_similar_in_pdf, data, body.page, body.bbox)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
