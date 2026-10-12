"""Text checks for the edge functions: grammar matches from the local LanguageTool server."""

import asyncio
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import verify_internal_access
from app.services.integrations import languagetool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/internal/text", tags=["Text (Internal)"])


class GrammarCheckRequest(BaseModel):
    language: str = Field(..., min_length=2, max_length=10, description="ISO code, e.g. 'el' or 'el-GR'")
    text: str = Field(..., min_length=1, max_length=languagetool.MAX_TEXT_CHARS)


@router.post("/grammar-check")
async def grammar_check(
    body: GrammarCheckRequest,
    claims: Optional[Dict[str, Any]] = Depends(verify_internal_access),
) -> Dict[str, Any]:
    try:
        result = await asyncio.to_thread(languagetool.check, body.language, body.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except languagetool.GrammarCheckUnavailable as exc:
        logger.warning("grammar-check unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"success": True, **result}
