"""
SAM (Segment Anything) Mask Generation Routes

Generates binary inpainting masks from zone hints.
Primary path: SAM 2 via Replicate (`meta/sam-2`) — pixel-perfect masks.
Fallback: Pillow bbox/ellipse — instant, zero cost.

Inpainting:
- AnyDoor (Replicate `ali-vilab/anydoor`) when `reference_image_url` provided — places real product photo.
- FLUX Fill Pro otherwise — text-guided replacement.
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
    image_url: Optional[str] = Field(default=None, description="Publicly accessible image URL (preferred — no base64 download)")
    image_base64: Optional[str] = Field(default=None, description="Base64-encoded source image (PNG or JPEG) — fallback if image_url not provided")
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
    method: str = Field(..., description="'sam2', 'bbox_fallback', or 'bbox' — indicates which method generated the mask")
    confidence: float = Field(..., description="Confidence of the mask (1.0 for bbox, variable for SAM 2)")


# ── SAM 2 via Replicate ───────────────────────────────────────────────────────

async def _generate_sam2_mask(
    image_url: str,
    bbox: BboxHint,
    img_w: int,
    img_h: int,
    replicate_token: str,
) -> Optional[str]:
    """
    Call meta/sam-2 on Replicate to get a pixel-perfect mask PNG.
    Returns base64-encoded PNG on success, None on failure.
    """
    box_x1 = int(bbox.x * img_w)
    box_y1 = int(bbox.y * img_h)
    box_x2 = int((bbox.x + bbox.w) * img_w)
    box_y2 = int((bbox.y + bbox.h) * img_h)

    payload = {
        "version": "meta/sam-2",
        "input": {
            "input_image": image_url,
            "box_x1": box_x1,
            "box_y1": box_y1,
            "box_x2": box_x2,
            "box_y2": box_y2,
        },
    }

    headers = {
        "Authorization": f"Bearer {replicate_token}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
        create_resp = await client.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=payload,
        )
        if create_resp.status_code not in (200, 201):
            logger.warning("SAM 2 prediction create failed: %s", create_resp.text)
            return None

        prediction_id = create_resp.json()["id"]
        logger.info("SAM 2 prediction created [id=%s, bbox=(%d,%d,%d,%d)]",
                    prediction_id, box_x1, box_y1, box_x2, box_y2)

        # Poll up to 60s
        for _ in range(20):
            await asyncio.sleep(3)
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
                mask_url = output[0] if isinstance(output, list) else output
                if not mask_url:
                    logger.warning("SAM 2 returned no output URL")
                    return None

                # Download the mask PNG
                dl = await client.get(mask_url, follow_redirects=True)
                if dl.status_code != 200:
                    logger.warning("Failed to download SAM 2 mask: %s", dl.status_code)
                    return None

                return base64.b64encode(dl.content).decode("utf-8")

            elif status == "failed":
                logger.warning("SAM 2 prediction failed: %s", result.get("error"))
                return None

    logger.warning("SAM 2 timed out")
    return None


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/sam", response_model=SAMMaskResponse)
async def generate_sam_mask(request: SAMMaskRequest) -> SAMMaskResponse:
    """
    Generate a binary inpainting mask from an image + zone hint.

    White pixels = area to replace (inpaint).
    Black pixels = area to preserve.

    Primary: SAM 2 (Replicate meta/sam-2) when `image_url` is provided + REPLICATE_API_TOKEN is set.
    Fallback: Pillow bbox/ellipse (instant, no external call).
    """
    try:
        from PIL import Image
        import numpy as np

        replicate_token = os.getenv("REPLICATE_API_TOKEN", "")

        # ── Determine image dimensions ─────────────────────────────────────
        if request.image_url and (request.image_width is None or request.image_height is None):
            # Fetch just enough bytes to decode dimensions without full download
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    head = await client.get(request.image_url, follow_redirects=True)
                    img_bytes = head.content
                img_pil = Image.open(io.BytesIO(img_bytes))
                src_w, src_h = img_pil.size
            except Exception:
                src_w, src_h = request.image_width or 1024, request.image_height or 768
        elif request.image_base64:
            image_bytes = base64.b64decode(request.image_base64)
            source_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            src_w, src_h = source_image.size
        else:
            src_w, src_h = request.image_width or 1024, request.image_height or 768

        out_w = request.image_width or src_w
        out_h = request.image_height or src_h

        # ── Try SAM 2 first (bbox hint + image_url) ────────────────────────
        if (
            replicate_token
            and request.image_url
            and request.hint_type == "bbox"
            and request.bbox
        ):
            try:
                sam2_b64 = await _generate_sam2_mask(
                    image_url=request.image_url,
                    bbox=request.bbox,
                    img_w=out_w,
                    img_h=out_h,
                    replicate_token=replicate_token,
                )
                if sam2_b64:
                    # Verify/normalise the mask dimensions
                    mask_img = Image.open(io.BytesIO(base64.b64decode(sam2_b64))).convert("L")
                    mw, mh = mask_img.size
                    logger.info("✅ SAM 2 mask [%dx%d]", mw, mh)
                    return SAMMaskResponse(
                        mask_base64=sam2_b64,
                        mask_width=mw,
                        mask_height=mh,
                        method="sam2",
                        confidence=0.92,
                    )
                logger.warning("SAM 2 returned None — falling back to Pillow bbox")
            except Exception as sam_err:
                logger.warning("SAM 2 failed (%s) — falling back to Pillow bbox", sam_err)

        # ── Pillow fallback ────────────────────────────────────────────────
        mask_array = np.zeros((out_h, out_w), dtype=np.uint8)

        if request.hint_type == "bbox" and request.bbox:
            bbox = request.bbox
            px = max(0, min(int(bbox.x * out_w), out_w - 1))
            py = max(0, min(int(bbox.y * out_h), out_h - 1))
            pw = max(1, min(int(bbox.w * out_w), out_w - px))
            ph = max(1, min(int(bbox.h * out_h), out_h - py))
            mask_array[py:py + ph, px:px + pw] = 255
            method = "bbox_fallback" if replicate_token and request.image_url else "bbox"
            confidence = 1.0

        elif request.hint_type == "point" and request.point:
            pt = request.point
            cx = int(pt.x * out_w)
            cy = int(pt.y * out_h)
            r_w = int(pt.radius * out_w)
            r_h = int(pt.radius * out_h)
            y_coords, x_coords = np.ogrid[:out_h, :out_w]
            ellipse_mask = ((x_coords - cx) / max(r_w, 1)) ** 2 + ((y_coords - cy) / max(r_h, 1)) ** 2 <= 1.0
            mask_array[ellipse_mask] = 255
            method = "bbox_fallback"
            confidence = 0.7

        else:
            raise HTTPException(
                status_code=422,
                detail="Provide either bbox (for hint_type='bbox') or point (for hint_type='point')"
            )

        mask_image = Image.fromarray(mask_array, mode="L")
        buffer = io.BytesIO()
        mask_image.save(buffer, format="PNG")
        mask_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        logger.info("SAM mask generated [method=%s, size=%dx%d]", method, out_w, out_h)

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

_ANYDOOR_VERSION = "ali-vilab/anydoor"


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
    reference_image_url: Optional[str] = Field(
        default=None,
        description="Catalog product photo URL — triggers AnyDoor reference-image inpainting instead of FLUX"
    )


class InpaintResponse(BaseModel):
    storage_url: str = Field(..., description="Supabase Storage public URL of the inpainted image")
    replicate_url: str = Field(..., description="Original Replicate output URL (may expire)")
    model_used: str
    processing_time_ms: int


@router.post("/inpaint", response_model=InpaintResponse)
async def inpaint_region(request: InpaintRequest) -> InpaintResponse:
    """
    Replace a masked region in an image.

    - If `reference_image_url` is provided → AnyDoor (places the actual product photo).
    - Otherwise → FLUX Fill Pro (text-guided generation).

    Mask must be a base64 PNG where white pixels = replace area, black = keep.
    Result is uploaded to Supabase Storage and a permanent URL is returned.
    """
    import time
    start = time.time()

    replicate_token = os.getenv("REPLICATE_API_TOKEN")
    if not replicate_token:
        raise HTTPException(status_code=500, detail="REPLICATE_API_TOKEN not configured")

    mask_data_url = f"data:image/png;base64,{request.mask_base64}"

    # ── AnyDoor path: reference-image inpainting ───────────────────────────
    if request.reference_image_url:
        payload = {
            "version": _ANYDOOR_VERSION,
            "input": {
                "bg_image": request.image_url,
                "bg_mask": mask_data_url,
                "ref_image": request.reference_image_url,
                "num_steps": 30,
                "guidance_scale": 3.5,
            },
        }
        model_label = "anydoor"

    # ── FLUX Fill Pro path: text-guided inpainting ─────────────────────────
    else:
        model_id = _INPAINT_MODELS.get(request.model, _INPAINT_MODELS["flux-fill-pro"])
        is_flux = request.model.startswith("flux")

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
        model_label = request.model

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            headers = {
                "Authorization": f"Bearer {replicate_token}",
                "Content-Type": "application/json",
            }

            create_resp = await client.post(
                "https://api.replicate.com/v1/predictions",
                headers=headers,
                json=payload,
            )
            if create_resp.status_code not in (200, 201):
                raise HTTPException(status_code=502, detail=f"Replicate API error: {create_resp.text}")

            prediction_id = create_resp.json()["id"]
            logger.info("Inpainting prediction created [id=%s, model=%s]", prediction_id, model_label)

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

                    storage_url = await _upload_to_storage(client, replicate_url, request.job_id)

                    processing_ms = int((time.time() - start) * 1000)
                    logger.info(
                        "Inpainting complete [job=%s, model=%s, %dms]",
                        request.job_id, model_label, processing_ms
                    )
                    return InpaintResponse(
                        storage_url=storage_url,
                        replicate_url=replicate_url,
                        model_used=model_label,
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


# ── Generate inpainting prompt via Claude (3c) ────────────────────────────────

class GeneratePromptRequest(BaseModel):
    zone_label: str = Field(..., description="Label of the zone being replaced (e.g. 'Floor', 'Wall')")
    description: str = Field(..., description="User description of desired material/look")
    zone_context: Optional[dict] = Field(default=None, description="Optional zone metadata (material_type, finish, etc.)")


class GeneratePromptResponse(BaseModel):
    prompt: str = Field(..., description="Claude-generated inpainting prompt")


@router.post("/generate-prompt", response_model=GeneratePromptResponse)
async def generate_inpainting_prompt(request: GeneratePromptRequest) -> GeneratePromptResponse:
    """
    Ask Claude Haiku to write an optimised FLUX Fill Pro inpainting prompt
    given a zone label and a user description.
    """
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")

    ctx_parts: list[str] = []
    if request.zone_context:
        if request.zone_context.get("material_type"):
            ctx_parts.append(f"current material: {request.zone_context['material_type']}")
        if request.zone_context.get("finish"):
            ctx_parts.append(f"finish: {request.zone_context['finish']}")
    ctx_note = f" ({', '.join(ctx_parts)})" if ctx_parts else ""

    user_message = (
        f"Write a technical inpainting prompt for FLUX Fill Pro to replace a {request.zone_label}{ctx_note} "
        f"in a photorealistic interior scene with the following look:\n\"{request.description}\"\n\n"
        "Include: material physics, perspective cues, lighting response (glossy/matte), seamless blending. "
        "Max 80 words. Return only the prompt text."
    )

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=200,
        messages=[{"role": "user", "content": user_message}],
    )
    prompt_text = message.content[0].text.strip()
    return GeneratePromptResponse(prompt=prompt_text)


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
