"""
Deriving a normal map from a material's albedo (#321 / #260 item 6).

WHY THIS IS NOT A GENERATION CALL. A diffusion model asked for "a normal map of this fabric"
returns an image that LOOKS like one — purple-blue, plausible — but a normal map is not a picture.
Each pixel's RGB encodes the surface direction at that point, and the renderer does arithmetic with
it. Invented directions do not correspond to the actual surface, so the lighting comes out wrong and
the material reads as cheap plastic. A bad normal map is worse than none.

So this derives one, deterministically, from the albedo the tenant already chose: luminance is
treated as height, the gradient of that height field gives the surface direction, and the direction
is encoded as RGB. That is the standard height-from-luminance approximation. It is an approximation
— a dark fabric is not a deep one — but it is an approximation OF THE REAL IMAGE rather than an
invention, it costs no credits, it is instant, and it is the same answer every time.

WHY HERE AND NOT IN AN EDGE FUNCTION. Supabase edge functions have no image decoding at all. MIVAA
already ships numpy, opencv and Pillow and already uses them, so this is a few lines against a stack
that exists rather than a new dependency anywhere.
"""

import io
import logging
from datetime import datetime
from typing import Any, Dict, Optional

import httpx
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_current_user, resolve_workspace_id
from app.utils.ssrf_guard import assert_safe_url, SSRFError
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/material-maps", tags=["material-maps"])

# Ceiling on the source image. A normal map is a texture, not a print: past this the extra pixels
# buy nothing a renderer can show and cost memory on a shared worker.
MAX_EDGE_PX = 2048


class NormalMapRequest(BaseModel):
    """Derive a normal map for one material-map row."""

    workspace_id: str = Field(..., description="Workspace that owns the product. Authorized, not trusted.")
    material_map_id: str = Field(..., description="product_material_maps row to derive from and write back to.")
    albedo_url: str = Field(..., description="Public URL of the albedo this is derived from.")
    strength: float = Field(
        1.0,
        ge=0.1,
        le=5.0,
        description=(
            "How pronounced the relief is. Above ~3 a weave starts to read as corrugation, which is "
            "why this is bounded rather than free."
        ),
    )


class NormalMapResponse(BaseModel):
    success: bool
    normal_path: Optional[str] = None
    error: Optional[str] = None


def derive_normal_map(image_bytes: bytes, strength: float = 1.0) -> bytes:
    """
    Height-from-luminance → surface normals → RGB.

    Kept as a plain function with no I/O so it can be reasoned about and tested directly: the route
    around it is fetch, call this, upload.
    """
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    if max(img.size) > MAX_EDGE_PX:
        scale = MAX_EDGE_PX / max(img.size)
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))

    height = np.asarray(img, dtype=np.float32) / 255.0

    # Gradients WRAP at the edges, because these textures tile. Using numpy's default edge handling
    # would put a visible seam exactly where the tile repeats — the one place it must not.
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5

    # The surface normal of a height field is (-dx, -dy, 1), scaled by how pronounced the relief is.
    nx = -dx * strength
    ny = -dy * strength
    nz = np.ones_like(height)

    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx, ny, nz = nx / length, ny / length, nz / length

    # Encode [-1, 1] into [0, 255]. A flat surface lands on (128, 128, 255) — the familiar lilac of
    # every normal map, and a useful smoke test: an output that is not mostly that colour is wrong.
    rgb = np.stack(
        [
            ((nx + 1.0) * 0.5 * 255.0),
            ((ny + 1.0) * 0.5 * 255.0),
            ((nz + 1.0) * 0.5 * 255.0),
        ],
        axis=-1,
    ).clip(0, 255).astype(np.uint8)

    out = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(out, format="PNG")
    return out.getvalue()


@router.post("/normal", response_model=NormalMapResponse)
async def generate_normal_map(
    request: NormalMapRequest,
    user: Dict[str, Any] = Depends(get_current_user),
) -> NormalMapResponse:
    """Derive a normal map from an albedo and record it on the material-map row."""
    # Invariant 1: the caller is authorized for the workspace they name; the id is never trusted
    # because it arrived in the body.
    await resolve_workspace_id(user, request.workspace_id)

    # Invariant 7: the URL is user-influenced and we fetch it, so it must not be able to reach an
    # internal address or the cloud metadata endpoint.
    try:
        assert_safe_url(request.albedo_url)
    except SSRFError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid albedo_url: {exc}")

    supabase = get_supabase_client()

    # The row must exist AND belong to the workspace the caller was authorized for. Without this a
    # member of workspace A could write a normal map onto workspace B's product by naming its id.
    row = (
        supabase.table("product_material_maps")
        .select("id, workspace_id, product_id, storage_bucket")
        .eq("id", request.material_map_id)
        .maybe_single()
        .execute()
    )
    record = getattr(row, "data", None)
    if not record or record.get("workspace_id") != request.workspace_id:
        # 404 rather than 403, so an id cannot be probed for existence.
        raise HTTPException(status_code=404, detail="Material map not found")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(request.albedo_url, follow_redirects=False)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Could not read the albedo ({resp.status_code})")
        normal_png = derive_normal_map(resp.content, request.strength)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 — the caller gets a reason, the log gets the trace
        logger.exception("normal map derivation failed")
        raise HTTPException(status_code=500, detail=f"Could not derive the normal map: {exc}")

    bucket = record.get("storage_bucket") or "generation-images"
    path = (
        f"material-maps/{record['product_id']}/normal-"
        f"{int(datetime.utcnow().timestamp() * 1000)}.png"
    )
    supabase.storage.from_(bucket).upload(
        path,
        normal_png,
        file_options={"content-type": "image/png", "upsert": "true"},
    )

    # A PATH, not a URL: the path is what build_storage_reference_set() registers, and therefore
    # what stops the nightly orphan cron reaping the file.
    supabase.table("product_material_maps").update({"normal_path": path}).eq(
        "id", request.material_map_id
    ).execute()

    return NormalMapResponse(success=True, normal_path=path)
