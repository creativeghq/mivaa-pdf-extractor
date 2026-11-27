"""Object → 3D: turn an object photo into a textured GLB mesh on fal and store it.

A product shot on a clean white background goes to Hunyuan 3D 3.1 Pro, which is sharper on a packshot;
anything else (a piece cut from a room render, a lifestyle photo) goes to SAM 3D Objects, the only one that
takes a mask or a noun saying WHICH object to build. The edge function `object-3d` owns auth, credits and
the claim on the library row; this route only runs the model and writes the result back onto that row.
"""

import asyncio
import io
import json
import logging
import struct
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.config.ai_pricing import AIPricingConfig
from app.dependencies import require_trusted_service
from app.modules._core.cost_logger import CostAttribution, log_external_call
from app.services.core.supabase_client import get_supabase_client
from app.services.integrations.platform_secret_resolver import require_secret

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/internal/object3d", tags=["object-3d"])

BUCKET = "generation-images"
FAL_QUEUE = "https://queue.fal.run"
DEADLINE_S = 600
POLL_S = 4
MAX_GLB_BYTES = 80 * 1024 * 1024
HUNYUAN_MAX_INPUT_BYTES = 8 * 1024 * 1024
HUNYUAN_FACES = 50000
_TASKS: set = set()


@dataclass(frozen=True)
class Engine:
    model_key: str
    endpoint: str
    label: str


SAM3D = Engine("sam-3d-objects", "fal-ai/sam-3/3d-objects", "SAM 3D")
HUNYUAN = Engine("hunyuan-3d-v31-pro", "fal-ai/hunyuan-3d/v3.1/pro/image-to-3d", "Hunyuan 3D")


class GenerateBody(BaseModel):
    target: Literal["product", "variant", "scene_object"]
    row_id: str
    image_path: str
    out_path: str
    mask_path: Optional[str] = None
    prompt: Optional[str] = None
    product_id: Optional[str] = None
    size_m: Optional[Dict[str, float]] = None
    refund: Optional[Dict[str, Any]] = None
    link_object_id: Optional[str] = None
    model_finish: Optional[str] = None
    clean_background: bool = False


def fal_input(image_url: str, mask_url: Optional[str], prompt: Optional[str], box: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """The SAM 3D body: a mask, else a box, else a prompt. fal segments "car" when given none, so that is refused."""
    body: Dict[str, Any] = {"image_url": image_url, "seed": 42, "export_textured_glb": True}
    if mask_url:
        body["mask_urls"] = [mask_url]
    elif box:
        body["box_prompts"] = [box]
    elif prompt and prompt.strip():
        body["prompt"] = prompt.strip()[:120]
    else:
        raise ValueError("A 3D model needs either a mask or a prompt naming the object.")
    return body


def hunyuan_input(image_url: str) -> Dict[str, Any]:
    return {"input_image_url": image_url, "face_count": HUNYUAN_FACES}


def foreground_box(image_bytes: bytes, white: int = 235) -> Optional[Dict[str, int]]:
    """Pixel box around everything that is not the white studio background of a clean shot."""
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    fg = img.point(lambda v: 255 if v < white else 0)
    bbox = fg.getbbox()
    if not bbox:
        return None
    x0, y0, x1, y1 = bbox
    return {"x_min": x0, "y_min": y0, "x_max": x1 - 1, "y_max": y1 - 1}


def is_clean_shot(image_bytes: bytes, white: int = 235, band: float = 0.04, share: float = 0.92) -> bool:
    """A packshot: the frame's outer band is studio white, so the photo holds the one object and nothing else."""
    from PIL import Image

    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    w, h = img.size
    bw, bh = max(1, int(w * band)), max(1, int(h * band))
    strips = [img.crop((0, 0, w, bh)), img.crop((0, h - bh, w, h)), img.crop((0, 0, bw, h)), img.crop((w - bw, 0, w, h))]
    total = sum(s.size[0] * s.size[1] for s in strips)
    bright = sum(sum(s.point(lambda v: 1 if v >= white else 0).histogram()[1:2]) for s in strips)
    has_object = foreground_box(image_bytes, white) is not None
    return has_object and bright / total >= share


def choose_engine(body: GenerateBody, image_bytes: Optional[bytes]) -> Engine:
    """Hunyuan for a product's clean shot; SAM 3D for room pieces and anything it must pick out of a scene."""
    if body.target == "scene_object" or body.mask_path or not image_bytes:
        return SAM3D
    if len(image_bytes) > HUNYUAN_MAX_INPUT_BYTES:
        return SAM3D
    return HUNYUAN if body.clean_background or is_clean_shot(image_bytes) else SAM3D


def slim_glb_textures(glb: bytes, max_px: int = 2048, quality: int = 88) -> bytes:
    """Re-encode opaque embedded textures as JPEG at most `max_px` wide: Hunyuan ships a 4096 PNG (~11 MB of a 12 MB file)."""
    from PIL import Image

    if glb[:4] != b"glTF":
        return glb
    jlen = struct.unpack("<I", glb[12:16])[0]
    doc = json.loads(glb[20:20 + jlen])
    pos = 20 + jlen
    blen = struct.unpack("<I", glb[pos:pos + 4])[0]
    bin_ = glb[pos + 8:pos + 8 + blen]
    views = doc.get("bufferViews") or []

    def view_bytes(v: Dict[str, Any]) -> bytes:
        start = v.get("byteOffset", 0)
        return bin_[start:start + v["byteLength"]]

    replaced: Dict[int, bytes] = {}
    for im in doc.get("images") or []:
        bv = im.get("bufferView")
        if bv is None or bv in replaced:
            continue
        raw = view_bytes(views[bv])
        img = Image.open(io.BytesIO(raw))
        img.load()
        if img.mode in ("RGBA", "LA", "PA") and img.getchannel("A").getextrema()[0] < 255:
            continue
        img = img.convert("RGB")
        if max(img.size) > max_px:
            img.thumbnail((max_px, max_px), Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, "JPEG", quality=quality, optimize=True)
        if out.tell() < len(raw):
            replaced[bv] = out.getvalue()
            im["mimeType"] = "image/jpeg"
    if not replaced:
        return glb
    chunks, offset = [], 0
    for i, v in enumerate(views):
        data = replaced.get(i) or view_bytes(v)
        pad = (-offset) % 4
        chunks.append(b"\0" * pad)
        offset += pad
        v["byteOffset"], v["byteLength"], v["buffer"] = offset, len(data), 0
        chunks.append(data)
        offset += len(data)
    new_bin = b"".join(chunks)
    new_bin += b"\0" * ((-len(new_bin)) % 4)
    doc["buffers"] = [{"byteLength": len(new_bin)}]
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(new_bin)
    return b"".join([
        b"glTF", struct.pack("<II", 2, total),
        struct.pack("<I", len(js)), b"JSON", js,
        struct.pack("<I", len(new_bin)), b"BIN\0", new_bin,
    ])


def stated_size(size: Optional[Dict[str, float]]) -> Dict[str, float]:
    """The product's real size in metres when a person stated one; a mesh from one photo has proportions, not scale."""
    try:
        dims = {f"{k}_m": float((size or {})[k]) for k in ("width", "depth", "height")}
    except (KeyError, TypeError, ValueError):
        return {}
    return dims if all(0.01 < v <= 20 for v in dims.values()) else {}


def is_fal_file_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return urlparse(url).scheme == "https" and (host == "fal.media" or host.endswith(".fal.media") or host.endswith(".fal.run"))


def unit_cost_usd(engine: Engine) -> float:
    row = AIPricingConfig._db_lookup(engine.model_key)
    if not row:
        raise RuntimeError(f"ai_model_pricing has no active '{engine.model_key}' row; the run cannot be priced.")
    return float(row["cost_per_unit"])


async def run_fal(client: httpx.AsyncClient, key: str, engine: Engine, payload: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    headers = {"Authorization": f"Key {key}"}
    sub = await client.post(f"{FAL_QUEUE}/{engine.endpoint}", json=payload, headers=headers)
    if sub.status_code != 200:
        raise RuntimeError(f"{engine.label} refused the job (HTTP {sub.status_code}): {sub.text[:300]}")
    job = sub.json()
    started = time.time()
    while True:
        st = await client.get(job["status_url"], headers=headers)
        if st.status_code not in (200, 202):
            raise RuntimeError(f"{engine.label} status check failed (HTTP {st.status_code}): {st.text[:300]}")
        if st.json().get("status") == "COMPLETED":
            break
        if time.time() - started > DEADLINE_S:
            raise RuntimeError(f"{engine.label} did not finish within {DEADLINE_S // 60} minutes.")
        await asyncio.sleep(POLL_S)
    res = await client.get(job["response_url"], headers=headers)
    if res.status_code != 200:
        raise RuntimeError(f"{engine.label} failed (HTTP {res.status_code}): {res.text[:300]}")
    return job.get("request_id", ""), res.json()


async def fetch_glb(client: httpx.AsyncClient, engine: Engine, out: Dict[str, Any]) -> bytes:
    url = ((out.get("model_glb") or {}).get("url")) or ""
    if not is_fal_file_url(url):
        raise RuntimeError(f"{engine.label} returned no mesh file.")
    res = await client.get(url, follow_redirects=False)
    if res.status_code != 200:
        raise RuntimeError(f"The mesh could not be downloaded (HTTP {res.status_code}).")
    data = res.content
    if len(data) > MAX_GLB_BYTES or data[:4] != b"glTF":
        raise RuntimeError(f"{engine.label} returned a file that is not a usable GLB mesh.")
    return data


def _mark(target: str, row_id: str, fields: Dict[str, Any]) -> None:
    sb = get_supabase_client().client
    if target == "product":
        sb.table("product_3d_models").update(fields).eq("id", row_id).execute()
    elif target == "variant":
        sb.table("product_3d_model_variants").update(fields).eq("id", row_id).execute()
    else:
        sb.table("floor_plan_objects").update(fields).eq("id", row_id).execute()


def _refund(refund: Optional[Dict[str, Any]], reason: str, target: str, row_id: str) -> None:
    if not refund or not refund.get("user_id") or not refund.get("credits"):
        return
    try:
        get_supabase_client().client.rpc("refund_credits", {
            "p_user_id": refund["user_id"],
            "p_amount": refund["credits"],
            "p_operation_type": "object_3d_refund",
            "p_description": f"3D generation failed: {reason[:120]}",
            "p_metadata": {"target": target, "row_id": row_id, "refunds_transaction_id": refund.get("transaction_id")},
            "p_workspace_id": refund.get("workspace_id"),
        }).execute()
    except Exception as e:  # noqa: BLE001
        logger.error("object3d refund failed for %s %s: %r", target, row_id, e)


def _log_run(body: GenerateBody, engine: Engine, cost: float, success: bool, wall_s: float, request_id: str, error: Optional[str]) -> None:
    refund = body.refund or {}
    log_external_call(
        module_slug="scene-studio",
        operation_type="object_3d_generation",
        model_name=engine.model_key,
        raw_cost_usd=cost,
        attribution=CostAttribution(
            subject_key="object_3d_row_id", subject_id=body.row_id,
            user_id=refund.get("user_id"), workspace_id=refund.get("workspace_id"), product_id=body.product_id,
        ),
        latency_ms=int(wall_s * 1000),
        extra_metadata={"target": body.target, "fal_request_id": request_id},
        success=success,
        error_message=error,
        provider_charged=success,
    )


async def _run(body: GenerateBody) -> None:
    started = time.time()
    request_id = ""
    cost = 0.0
    engine = SAM3D
    try:
        key = require_secret("FAL_KEY")
        store = get_supabase_client().client.storage.from_(BUCKET)
        image = None if body.target == "scene_object" and body.mask_path else await asyncio.to_thread(store.download, body.image_path)
        engine = choose_engine(body, image)
        cost = unit_cost_usd(engine)
        image_url = store.get_public_url(body.image_path)
        if engine is HUNYUAN:
            payload = hunyuan_input(image_url)
        else:
            box = foreground_box(image) if body.clean_background and not body.mask_path and image else None
            payload = fal_input(image_url, store.get_public_url(body.mask_path) if body.mask_path else None, body.prompt, box)
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
            request_id, out = await run_fal(client, key, engine, payload)
            raw = await fetch_glb(client, engine, out)
        glb = await asyncio.to_thread(slim_glb_textures, raw)
        await asyncio.to_thread(store.upload, body.out_path, glb, {"content-type": "model/gltf-binary", "upsert": "true"})
        wall_s = round(time.time() - started, 1)
        _log_run(body, engine, cost, True, wall_s, request_id, None)
        quality = {
            "bytes": len(glb), "raw_bytes": len(raw), "fal_request_id": request_id,
            "objects": len(out.get("metadata") or []), "wall_s": wall_s,
        }
        now = datetime.now(timezone.utc).isoformat()
        if body.target in ("product", "variant"):
            _mark(body.target, body.row_id, {
                "status": "ready", "storage_path": body.out_path, "file_size_bytes": len(glb),
                "generator": engine.model_key, "generator_version": engine.endpoint, "quality": quality, "error": None,
                **(stated_size(body.size_m) if body.target == "product" else {}),
            })
            if body.link_object_id:
                _mark("scene_object", body.link_object_id, {
                    "model_status": "ready", "model_storage_path": body.out_path, "model_error": None,
                    "model_finish": body.model_finish, "model_updated_at": now,
                })
        else:
            _mark("scene_object", body.row_id, {
                "model_status": "ready", "model_storage_path": body.out_path, "model_error": None,
                "model_finish": body.model_finish, "model_updated_at": now,
            })
        logger.info("object3d %s %s ready via %s %s", body.target, body.row_id, engine.model_key, quality)
    except Exception as e:  # noqa: BLE001
        logger.warning("object3d %s %s failed: %r", body.target, body.row_id, e)
        msg = str(e)[:500]
        _log_run(body, engine, cost, False, time.time() - started, request_id, msg)
        _refund(body.refund, msg, body.target, body.row_id)
        if body.target in ("product", "variant"):
            _mark(body.target, body.row_id, {"status": "failed", "error": msg})
            if body.link_object_id:
                _mark("scene_object", body.link_object_id, {"model_status": "failed", "model_error": msg})
        else:
            _mark("scene_object", body.row_id, {"model_status": "failed", "model_error": msg})


def _safe_path(path: Optional[str]) -> bool:
    return path is None or (bool(path) and ".." not in path and not path.startswith("/"))


@router.post("/generate", status_code=202)
async def generate(body: GenerateBody, _caller: Optional[Dict[str, Any]] = Depends(require_trusted_service)):
    if not body.image_path or not all(_safe_path(p) for p in (body.image_path, body.out_path, body.mask_path)):
        raise HTTPException(status_code=400, detail="bad path")
    if not body.out_path.endswith(".glb"):
        raise HTTPException(status_code=400, detail="out_path must be a .glb")
    task = asyncio.create_task(_run(body))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return {"accepted": True}
