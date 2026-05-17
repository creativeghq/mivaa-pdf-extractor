"""
Catalog rasterization helper for the presentation_catalogs flow.

Single endpoint: POST /api/internal/catalog/rasterize-pdf-page

Renders one page of an admin-uploaded source PDF (stored in the
`catalog-sources` Supabase Storage bucket) to PNG, optionally cropped to a
normalized [0..1] bounding box, and uploads the result to
`catalog-extracted-images/<source_pdf_id>/page-<n>[-<bbox-hash>].png`.

Used by the catalog-extract / catalog-translate edge functions to give every
extracted material a real image lifted from the source PDF, instead of leaving
image_url=null and asking the admin to find one.

Auth: x-cron-secret header (same pattern as the other internal endpoints).
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
from typing import Optional

import fitz  # PyMuPDF — already installed for the main pipeline
from fastapi import APIRouter, HTTPException, Request
from PIL import Image
from pydantic import BaseModel, Field

from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/internal/catalog",
    tags=["Catalog (Internal)"],
    responses={
        401: {"description": "Bad cron secret"},
        404: {"description": "Source PDF not found"},
        422: {"description": "Invalid arguments"},
    },
)


def _check_secret(request: Request) -> None:
    secret = request.headers.get("x-cron-secret")
    expected = os.getenv("CRON_SECRET")
    if not expected or secret != expected:
        raise HTTPException(status_code=401, detail="bad cron secret")


class BBox(BaseModel):
    """Normalized [0..1] bounding box. (0,0) = top-left of the page."""
    x1: float = Field(..., ge=0.0, le=1.0)
    y1: float = Field(..., ge=0.0, le=1.0)
    x2: float = Field(..., ge=0.0, le=1.0)
    y2: float = Field(..., ge=0.0, le=1.0)


class RasterizeRequest(BaseModel):
    source_pdf_id: str
    page_no: int = Field(..., ge=1, description="1-based PDF page number")
    bbox: Optional[BBox] = None
    dpi: int = Field(200, ge=72, le=400, description="Render DPI. Higher = sharper but heavier.")
    target_path: Optional[str] = Field(
        None,
        description="Override storage path under catalog-extracted-images. Default is auto-derived.",
    )
    signed_url_ttl_seconds: int = Field(60 * 60 * 24 * 7)  # 7 days


class RasterizeResponse(BaseModel):
    success: bool
    image_url: Optional[str] = None
    storage_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    error: Optional[str] = None


def _bbox_hash(bbox: Optional[BBox]) -> str:
    if not bbox:
        return "full"
    raw = f"{bbox.x1:.4f}_{bbox.y1:.4f}_{bbox.x2:.4f}_{bbox.y2:.4f}"
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


@router.post("/rasterize-pdf-page", response_model=RasterizeResponse)
async def rasterize_pdf_page(payload: RasterizeRequest, request: Request) -> RasterizeResponse:
    """Render one page (optionally cropped) of a source PDF as PNG and upload it."""
    _check_secret(request)
    supabase = get_supabase_client()

    pdf_row = (
        supabase.client.table("catalog_source_pdfs")
        .select("id, storage_path, original_filename")
        .eq("id", payload.source_pdf_id)
        .maybe_single()
        .execute()
    )
    if not pdf_row or not pdf_row.data:
        raise HTTPException(status_code=404, detail=f"source_pdf {payload.source_pdf_id} not found")

    storage_path = pdf_row.data["storage_path"]

    try:
        pdf_bytes = supabase.client.storage.from_("catalog-sources").download(storage_path)
    except Exception as e:
        logger.exception("[catalog/rasterize] download failed")
        raise HTTPException(status_code=500, detail=f"download failed: {e}")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        logger.exception("[catalog/rasterize] open failed")
        raise HTTPException(status_code=422, detail=f"invalid PDF: {e}")

    try:
        if payload.page_no > len(doc):
            raise HTTPException(
                status_code=422,
                detail=f"page_no {payload.page_no} > total pages {len(doc)}",
            )

        page = doc[payload.page_no - 1]
        zoom = payload.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        png_bytes = pix.tobytes("png")

        if payload.bbox is not None:
            img = Image.open(io.BytesIO(png_bytes))
            w, h = img.size
            left = max(0, int(payload.bbox.x1 * w))
            top = max(0, int(payload.bbox.y1 * h))
            right = min(w, int(payload.bbox.x2 * w))
            bottom = min(h, int(payload.bbox.y2 * h))
            if right > left and bottom > top:
                img = img.crop((left, top, right, bottom))
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                png_bytes = buf.getvalue()
                width, height = img.size
            else:
                width, height = pix.width, pix.height
                logger.warning(
                    "[catalog/rasterize] degenerate bbox %s for page (%dx%d) — falling back to full page",
                    payload.bbox.dict(), w, h,
                )
        else:
            width, height = pix.width, pix.height

        target = payload.target_path or (
            f"{payload.source_pdf_id}/page-{payload.page_no:04d}-{_bbox_hash(payload.bbox)}.png"
        )

        try:
            supabase.client.storage.from_("catalog-extracted-images").upload(
                target,
                png_bytes,
                {"content-type": "image/png", "upsert": "true"},
            )
        except Exception:
            try:
                supabase.client.storage.from_("catalog-extracted-images").update(
                    target, png_bytes, {"content-type": "image/png"},
                )
            except Exception as e:
                logger.exception("[catalog/rasterize] upload failed")
                raise HTTPException(status_code=500, detail=f"upload failed: {e}")

        signed = supabase.client.storage.from_("catalog-extracted-images").create_signed_url(
            target, payload.signed_url_ttl_seconds
        )
        signed_url = (signed or {}).get("signedURL") or (signed or {}).get("signed_url")

        return RasterizeResponse(
            success=True,
            image_url=signed_url,
            storage_path=target,
            width=width,
            height=height,
        )
    finally:
        doc.close()
