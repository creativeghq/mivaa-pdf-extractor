"""
SAM (Segment Anything) Mask Generation Routes

Generates binary inpainting masks from zone hints.
Primary path: bbox hint → Pillow-generated rectangular mask (instant, zero cost).
The frontend may call Replicate SAM 2 for pixel-level masks and pass the result
directly to the inpainting model — this endpoint serves as the reliable fallback.
"""

import base64
import io
import logging
import os
import asyncio
from datetime import datetime
from typing import Literal, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/segment", tags=["Segmentation"])


# ── Request / Response schemas ────────────────────────────────────────────────

class BboxHint(BaseModel):
    """Normalized bbox (0.0–1.0)."""
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    w: float = Field(..., ge=0.0, le=1.0)
    h: float = Field(..., ge=0.0, le=1.0)


class PointHint(BaseModel):
    """Normalized point (0.0–1.0). Generates a small default region around the click."""
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    radius: float = Field(default=0.15, ge=0.01, le=0.5, description="Radius as fraction of image size")


class SAMMaskRequest(BaseModel):
    image_base64: str = Field(..., description="Base64-encoded source image (PNG or JPEG)")
    hint_type: Literal["bbox", "point"] = Field(default="bbox")
    bbox: Optional[BboxHint] = None
    point: Optional[PointHint] = None
    image_width: Optional[int] = Field(default=None, description="Output mask width in px; defaults to source image width")
    image_height: Optional[int] = Field(default=None, description="Output mask height in px; defaults to source image height")
    workspace_id: Optional[str] = None


class SAMMaskResponse(BaseModel):
    mask_base64: str = Field(..., description="Base64-encoded PNG mask (white=replace, black=keep)")
    mask_width: int
    mask_height: int
    method: str = Field(..., description="'bbox' or 'sam2' — indicates which method generated the mask")
    confidence: float = Field(..., description="Confidence of the mask (1.0 for bbox, variable for SAM)")


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/sam", response_model=SAMMaskResponse)
async def generate_sam_mask(request: SAMMaskRequest) -> SAMMaskResponse:
    """
    Generate a binary inpainting mask from an image + zone hint.

    White pixels = area to replace (inpaint).
    Black pixels = area to preserve.

    Uses Pillow for bbox-based masks (instant). For pixel-precise masks,
    the frontend should call Replicate SAM 2 directly and use that result.
    """
    try:
        from PIL import Image
        import numpy as np

        # Decode source image to determine natural dimensions
        image_bytes = base64.b64decode(request.image_base64)
        source_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        src_w, src_h = source_image.size

        # Use provided dimensions or fall back to source image dimensions
        out_w = request.image_width or src_w
        out_h = request.image_height or src_h

        # Create black canvas (everything preserved by default)
        mask_array = np.zeros((out_h, out_w), dtype=np.uint8)

        if request.hint_type == "bbox" and request.bbox:
            bbox = request.bbox
            # Convert normalized coords → pixel coords
            px = int(bbox.x * out_w)
            py = int(bbox.y * out_h)
            pw = int(bbox.w * out_w)
            ph = int(bbox.h * out_h)

            # Clamp to image bounds
            px = max(0, min(px, out_w - 1))
            py = max(0, min(py, out_h - 1))
            pw = max(1, min(pw, out_w - px))
            ph = max(1, min(ph, out_h - py))

            # Paint white rectangle (area to replace)
            mask_array[py:py + ph, px:px + pw] = 255
            method = "bbox"
            confidence = 1.0

        elif request.hint_type == "point" and request.point:
            pt = request.point
            cx = int(pt.x * out_w)
            cy = int(pt.y * out_h)
            r_w = int(pt.radius * out_w)
            r_h = int(pt.radius * out_h)

            # Elliptical region around the click point
            y_coords, x_coords = np.ogrid[:out_h, :out_w]
            ellipse_mask = ((x_coords - cx) / max(r_w, 1)) ** 2 + ((y_coords - cy) / max(r_h, 1)) ** 2 <= 1.0
            mask_array[ellipse_mask] = 255
            method = "bbox"  # still bbox-class (no real SAM)
            confidence = 0.7

        else:
            raise HTTPException(
                status_code=422,
                detail="Provide either bbox (for hint_type='bbox') or point (for hint_type='point')"
            )

        # Encode mask as PNG base64
        mask_image = Image.fromarray(mask_array, mode="L")
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        mask_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        logger.info(
            "SAM mask generated [method=%s, size=%dx%d, hint=%s]",
            method, out_w, out_h, request.hint_type
        )

        return SAMMaskResponse(
            mask_base64=mask_b64,
            mask_width=out_w,
            mask_height=out_h,
            method=method,
            confidence=confidence,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("SAM mask generation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Mask generation failed: {str(e)}")


# ── Inpainting endpoint ───────────────────────────────────────────────────────

_INPAINT_MODELS = {
    "flux-fill-pro": "black-forest-labs/flux-fill-pro",
    "flux-fill-dev": "black-forest-labs/flux-fill-dev",
    "sd-inpainting": "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3",
}


class InpaintRequest(BaseModel):
    image_url: str = Field(..., description="Source image URL (must be publicly accessible)")
    mask_base64: str = Field(..., description="Base64-encoded PNG mask (white=replace, black=keep)")
    prompt: str = Field(..., description="Inpainting prompt describing the replacement material")
    negative_prompt: str = Field(
        default="blurry, artifacts, distorted, low quality, unrealistic, cartoon, illustration, collage edges",
    )
    model: Literal["flux-fill-pro", "flux-fill-dev", "sd-inpainting"] = Field(default="flux-fill-pro")
    job_id: str = Field(..., description="Parent generation job ID — used for Storage path")
    workspace_id: Optional[str] = None


class InpaintResponse(BaseModel):
    storage_url: str = Field(..., description="Supabase Storage public URL of the inpainted image")
    replicate_url: str = Field(..., description="Original Replicate output URL (may expire)")
    model_used: str
    processing_time_ms: int


@router.post("/inpaint", response_model=InpaintResponse)
async def inpaint_region(request: InpaintRequest) -> InpaintResponse:
    """
    Replace a masked region in an image using FLUX Fill Pro (default) or other models.

    Mask must be a base64 PNG where white pixels = replace area, black = keep.
    Result is uploaded to Supabase Storage and a permanent URL is returned.
    """
    import time
    start = time.time()

    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    if not replicate_token:
        raise HTTPException(status_code=500, detail="REPLICATE_API_TOKEN not configured")

    model_id = _INPAINT_MODELS.get(request.model, _INPAINT_MODELS["flux-fill-pro"])
    is_flux = request.model.startswith("flux")

    # Build mask data URL for Replicate
    mask_data_url = f"data:image/png;base64,{request.mask_base64}"

    if is_flux:
        payload = {
            "model": model_id,
            "input": {
                "image": request.image_url,
                "mask": mask_data_url,
                "prompt": request.prompt,
            },
        }
    else:
        payload = {
            "version": model_id,
            "input": {
                "image": request.image_url,
                "mask": mask_data_url,
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
            },
        }

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            headers = {
                "Authorization": f"Bearer {replicate_token}",
                "Content-Type": "application/json",
            }

            # Create prediction
            create_resp = await client.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json=payload,
            )
            if create_resp.status_code not in (200, 201):
                raise HTTPException(status_code=502, detail=f"Replicate API error: {create_resp.text}")

            prediction_id = create_resp.json()["id"]
            logger.info("Inpainting prediction created [id=%s, model=%s]", prediction_id, request.model)

            # Poll for completion (up to 5 minutes)
            for _ in range(90):
                await asyncio.sleep(4)
                poll_resp = await client.get(
                    f"https://api.replicate.com/v1/predictions/{prediction_id}",
                    headers=headers,
                )
                if poll_resp.status_code != 200:
                    continue

                result = poll_resp.json()
                status = result.get("status")

                if status == "succeeded":
                    output = result.get("output")
                    replicate_url = output[0] if isinstance(output, list) else output
                    if not replicate_url:
                        raise HTTPException(status_code=500, detail="Inpainting returned no output URL")

                    # Download and upload to Supabase Storage
                    storage_url = await _upload_to_storage(client, replicate_url, request.job_id)

                    processing_ms = int((time.time() - start) * 1000)
                    logger.info(
                        "Inpainting complete [job=%s, model=%s, %dms]",
                        request.job_id, request.model, processing_ms
                    )
                    return InpaintResponse(
                        storage_url=storage_url,
                        replicate_url=replicate_url,
                        model_used=request.model,
                        processing_time_ms=processing_ms,
                    )

                elif status == "failed":
                    raise HTTPException(
                        status_code=500,
                        detail=f"Inpainting failed: {result.get('error', 'Unknown error')}"
                    )

            raise HTTPException(status_code=504, detail="Inpainting timed out after 5 minutes")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Inpainting error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inpainting error: {str(e)}")


async def _upload_to_storage(client: httpx.AsyncClient, image_url: str, job_id: str) -> str:
    """Download a Replicate URL and upload to Supabase Storage. Returns the public URL."""
    dl_resp = await client.get(image_url, follow_redirects=True)
    if dl_resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Failed to download inpainted image: {dl_resp.status_code}")

    content_type = dl_resp.headers.get("content-type", "image/webp")
    ext = "png" if "png" in content_type else "jpg"
    path = f"generation-edits/{job_id}/{int(datetime.utcnow().timestamp() * 1000)}.{ext}"

    supabase = get_supabase_client()
    supabase.storage.from_("product-images").upload(
        path,
        dl_resp.content,
        file_options={"content-type": content_type, "upsert": "true"},
    )

    url_data = supabase.storage.from_("product-images").get_public_url(path)
    return url_data
