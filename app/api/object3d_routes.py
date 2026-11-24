"""Object → 3D: turn a clean single-object image into a Gaussian splat with Idol (Modal) and store it.

The edge function `object-3d` owns auth, credits and the claim on the library row; this route only
does the GPU work and writes the result back onto that row.
"""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

import httpx
import numpy as np
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.dependencies import require_trusted_service
from app.services.core.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/internal/object3d", tags=["object-3d"])

BUCKET = "generation-images"
IDOL_DEFAULT_URL = "https://basilakis--idol-idolservice-web.us-east.modal.run"
IDOL_SDK = "8db700349a5570aac390e4780fa32e3eed6b587a"
_TASKS: set = set()


class GenerateBody(BaseModel):
    target: Literal["product", "variant", "scene_object"]
    row_id: str
    image_path: str
    out_path: str
    refund: Optional[Dict[str, Any]] = None
    link_object_id: Optional[str] = None
    model_finish: Optional[str] = None


def _idol_endpoint() -> tuple[str, str]:
    key = os.getenv("IDOL_MODAL_API_KEY") or os.getenv("SLIG_MODAL_API_KEY") or ""
    return os.getenv("IDOL_MODAL_URL", IDOL_DEFAULT_URL), key


def _read_ply(data: bytes):
    end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:end].decode()
    props = [ln.split()[-1] for ln in header.splitlines() if ln.startswith("property")]
    count = int(next(ln for ln in header.splitlines() if ln.startswith("element vertex")).split()[-1])
    arr = np.frombuffer(data[end:], dtype=np.float32, count=count * len(props)).reshape(count, len(props))
    return header, props, arr


def _write_ply(header: str, arr: np.ndarray) -> bytes:
    lines = [f"element vertex {len(arr)}" if ln.startswith("element vertex") else ln for ln in header.splitlines()]
    return ("\n".join(lines) + "\n").encode() + arr.astype(np.float32).tobytes()


def clean_splat(data: bytes) -> tuple[bytes, Dict[str, int]]:
    """Drop near-invisible Gaussians and detached specks; keep every piece at least 3% the size of the main one.

    Keeping all sizeable pieces, not only the largest, is what keeps thin parts (lamp legs) that a
    coarse grid sees as separate from the body."""
    from scipy import ndimage

    header, props, arr = _read_ply(data)
    xyz = arr[:, [props.index("x"), props.index("y"), props.index("z")]]
    opacity = 1 / (1 + np.exp(-arr[:, props.index("opacity")]))
    keep = opacity > 0.04
    if keep.sum() < 1000:
        return data, {"total": len(arr), "kept": len(arr)}
    lo, hi = xyz[keep].min(0), xyz[keep].max(0)
    cell = float((hi - lo).max()) / 128
    idx = np.floor((xyz - lo) / cell).astype(int)
    shape = tuple(int(v) + 1 for v in idx[keep].max(0))
    grid = np.zeros(shape, dtype=bool)
    k = idx[keep]
    grid[k[:, 0], k[:, 1], k[:, 2]] = True
    grid = ndimage.binary_dilation(grid, iterations=2)
    labels, n = ndimage.label(grid)
    if n > 1:
        sizes = ndimage.sum(grid, labels, range(1, n + 1))
        big = {i + 1 for i, s in enumerate(sizes) if s >= 0.03 * sizes.max()}
        inside = np.zeros(len(arr), dtype=bool)
        ok = keep & np.all(idx >= 0, axis=1) & np.all(idx < np.array(shape), axis=1)
        inside[ok] = np.isin(labels[idx[ok, 0], idx[ok, 1], idx[ok, 2]], list(big))
        keep = keep & inside
    return _write_ply(header, arr[keep]), {"total": int(len(arr)), "kept": int(keep.sum())}


SH_C0 = 0.28209479177387814


def to_spz(data: bytes, fractional_bits: int = 12) -> bytes:
    """A degree-0 splat PLY as SPZ v3, byte-for-byte what Spark's SpzWriter writes (about 7x smaller)."""
    import gzip

    _, props, arr = _read_ply(data)
    col = lambda name: arr[:, props.index(name)].astype(np.float64)  # noqa: E731
    n = len(arr)
    pos = np.stack([col("x"), col("y"), col("z")], 1)
    fixed = np.clip(np.floor(0.5 + pos * (1 << fractional_bits)), -8388607, 8388607).astype(np.int32)
    xyz = np.stack([(fixed >> s) & 255 for s in (0, 8, 16)], 2).astype(np.uint8).reshape(n, 9)
    alpha = np.clip(np.floor(0.5 + 255 / (1 + np.exp(-col("opacity")))), 0, 255).astype(np.uint8)
    rgb = np.stack([col(f"f_dc_{i}") for i in range(3)], 1)
    rgb = np.clip(np.floor(0.5 + (rgb * SH_C0 / (SH_C0 / 0.15) + 0.5) * 255), 0, 255).astype(np.uint8)
    scale = np.clip(np.floor(0.5 + (np.stack([col(f"scale_{i}") for i in range(3)], 1) + 10) * 16), 0, 255).astype(np.uint8)
    q = np.stack([col("rot_1"), col("rot_2"), col("rot_3"), col("rot_0")], 1)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    largest = np.argmax(np.abs(q), 1)
    negate = q[np.arange(n), largest] < 0
    comp = largest.astype(np.uint64)
    for i in range(4):
        other = largest != i
        neg = ((q[:, i] < 0) ^ negate).astype(np.uint64)
        mag = np.floor(511 * (np.abs(q[:, i]) / np.sqrt(0.5)) + 0.5).astype(np.uint64)
        comp = np.where(other, (comp << np.uint64(10)) | (neg << np.uint64(9)) | mag, comp)
    rot = np.stack([(comp >> np.uint64(s)) & np.uint64(255) for s in (0, 8, 16, 24)], 1).astype(np.uint8)
    header = np.array([1347635022, 3, n], dtype="<u4").tobytes() + bytes([0, fractional_bits, 1, 0])
    body = xyz.tobytes() + alpha.tobytes() + rgb.tobytes() + scale.tobytes() + rot.tobytes()
    return gzip.compress(header + body, compresslevel=9)


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


async def _run(body: GenerateBody) -> None:
    started = time.time()
    try:
        sb = get_supabase_client().client
        image = await asyncio.to_thread(sb.storage.from_(BUCKET).download, body.image_path)
        url, key = _idol_endpoint()
        if not key:
            raise RuntimeError("IDOL_MODAL_API_KEY / SLIG_MODAL_API_KEY is not configured")
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
            res = await client.post(
                f"{url}/generate",
                files={"image": ("object.png", image, "image/png")},
                data={"mode": "synthetic", "num_gaussians": "262144", "seed": "42"},
                headers={"Authorization": f"Bearer {key}"},
            )
        if res.status_code != 200:
            raise RuntimeError(f"Idol HTTP {res.status_code}: {res.text[:300]}")
        cleaned, stats = await asyncio.to_thread(clean_splat, res.content)
        spz = body.out_path.endswith(".spz")
        if spz:
            cleaned = await asyncio.to_thread(to_spz, cleaned)
        await asyncio.to_thread(
            sb.storage.from_(BUCKET).upload, body.out_path, cleaned,
            {"content-type": "application/x-spz" if spz else "application/x-ply", "upsert": "true"},
        )
        quality = {**stats, "gpu_s": res.headers.get("X-Idol-Seconds"), "wall_s": round(time.time() - started, 1)}
        if body.target in ("product", "variant"):
            _mark(body.target, body.row_id, {
                "status": "ready", "storage_path": body.out_path, "file_size_bytes": len(cleaned),
                "generator": "idol", "generator_version": IDOL_SDK, "quality": quality, "error": None,
            })
            if body.link_object_id:
                _mark("scene_object", body.link_object_id, {
                    "model_status": "ready", "model_storage_path": body.out_path, "model_error": None,
                    "model_finish": body.model_finish, "model_updated_at": datetime.now(timezone.utc).isoformat(),
                })
        else:
            _mark("scene_object", body.row_id, {
                "model_status": "ready", "model_storage_path": body.out_path, "model_error": None,
                "model_finish": body.model_finish, "model_updated_at": datetime.now(timezone.utc).isoformat(),
            })
        logger.info("object3d %s %s ready %s", body.target, body.row_id, quality)
    except Exception as e:  # noqa: BLE001
        logger.warning("object3d %s %s failed: %r", body.target, body.row_id, e)
        msg = str(e)[:500]
        _refund(body.refund, msg, body.target, body.row_id)
        if body.target in ("product", "variant"):
            _mark(body.target, body.row_id, {"status": "failed", "error": msg})
            if body.link_object_id:
                _mark("scene_object", body.link_object_id, {"model_status": "failed", "model_error": msg})
        else:
            _mark("scene_object", body.row_id, {"model_status": "failed", "model_error": msg})


@router.post("/generate", status_code=202)
async def generate(body: GenerateBody, _caller: Optional[Dict[str, Any]] = Depends(require_trusted_service)):
    if not body.image_path or ".." in body.image_path or ".." in body.out_path:
        return {"accepted": False, "error": "bad path"}
    task = asyncio.create_task(_run(body))
    _TASKS.add(task)
    task.add_done_callback(_TASKS.discard)
    return {"accepted": True}
