"""Object → 3D: turn an object photo into a textured GLB mesh with SAM 3D Objects (fal) and store it.

The edge function `object-3d` owns auth, credits and the claim on the library row; this route only
runs the model and writes the result back onto that row.
"""

import asyncio
import logging
import time
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
FAL_ENDPOINT = "fal-ai/sam-3/3d-objects"
MODEL_KEY = "sam-3d-objects"
DEADLINE_S = 600
POLL_S = 4
MAX_GLB_BYTES = 80 * 1024 * 1024
_TASKS: set = set()


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


def fal_input(image_url: str, mask_url: Optional[str], prompt: Optional[str]) -> Dict[str, Any]:
    """The request body. fal segments "car" when given neither a mask nor a prompt, so that is refused."""
    body: Dict[str, Any] = {"image_url": image_url, "seed": 42, "export_textured_glb": True}
    if mask_url:
        body["mask_urls"] = [mask_url]
    elif prompt and prompt.strip():
        body["prompt"] = prompt.strip()[:120]
    else:
        raise ValueError("A 3D model needs either a mask or a prompt naming the object.")
    return body


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


def unit_cost_usd() -> float:
    row = AIPricingConfig._db_lookup(MODEL_KEY)
    if not row:
        raise RuntimeError(f"ai_model_pricing has no active '{MODEL_KEY}' row; the run cannot be priced.")
    return float(row["cost_per_unit"])


async def run_fal(client: httpx.AsyncClient, key: str, payload: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    headers = {"Authorization": f"Key {key}"}
    sub = await client.post(f"{FAL_QUEUE}/{FAL_ENDPOINT}", json=payload, headers=headers)
    if sub.status_code != 200:
        raise RuntimeError(f"SAM 3D refused the job (HTTP {sub.status_code}): {sub.text[:300]}")
    job = sub.json()
    started = time.time()
    while True:
        st = await client.get(job["status_url"], headers=headers)
        if st.status_code not in (200, 202):
            raise RuntimeError(f"SAM 3D status check failed (HTTP {st.status_code}): {st.text[:300]}")
        if st.json().get("status") == "COMPLETED":
            break
        if time.time() - started > DEADLINE_S:
            raise RuntimeError(f"SAM 3D did not finish within {DEADLINE_S // 60} minutes.")
        await asyncio.sleep(POLL_S)
    res = await client.get(job["response_url"], headers=headers)
    if res.status_code != 200:
        raise RuntimeError(f"SAM 3D failed (HTTP {res.status_code}): {res.text[:300]}")
    return job.get("request_id", ""), res.json()


async def fetch_glb(client: httpx.AsyncClient, out: Dict[str, Any]) -> bytes:
    url = ((out.get("model_glb") or {}).get("url")) or ""
    if not is_fal_file_url(url):
        raise RuntimeError("SAM 3D returned no mesh file.")
    res = await client.get(url, follow_redirects=False)
    if res.status_code != 200:
        raise RuntimeError(f"The mesh could not be downloaded (HTTP {res.status_code}).")
    data = res.content
    if len(data) > MAX_GLB_BYTES or data[:4] != b"glTF":
        raise RuntimeError("SAM 3D returned a file that is not a usable GLB mesh.")
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


def _log_run(body: GenerateBody, cost: float, success: bool, wall_s: float, request_id: str, error: Optional[str]) -> None:
    refund = body.refund or {}
    log_external_call(
        module_slug="scene-studio",
        operation_type="object_3d_generation",
        model_name=MODEL_KEY,
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
    try:
        cost = unit_cost_usd()
        key = require_secret("FAL_KEY")
        store = get_supabase_client().client.storage.from_(BUCKET)
        payload = fal_input(
            store.get_public_url(body.image_path),
            store.get_public_url(body.mask_path) if body.mask_path else None,
            body.prompt,
        )
        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
            request_id, out = await run_fal(client, key, payload)
            glb = await fetch_glb(client, out)
        await asyncio.to_thread(store.upload, body.out_path, glb, {"content-type": "model/gltf-binary", "upsert": "true"})
        wall_s = round(time.time() - started, 1)
        _log_run(body, cost, True, wall_s, request_id, None)
        quality = {"bytes": len(glb), "fal_request_id": request_id, "objects": len(out.get("metadata") or []), "wall_s": wall_s}
        now = datetime.now(timezone.utc).isoformat()
        if body.target in ("product", "variant"):
            _mark(body.target, body.row_id, {
                "status": "ready", "storage_path": body.out_path, "file_size_bytes": len(glb),
                "generator": MODEL_KEY, "generator_version": FAL_ENDPOINT, "quality": quality, "error": None,
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
        logger.info("object3d %s %s ready %s", body.target, body.row_id, quality)
    except Exception as e:  # noqa: BLE001
        logger.warning("object3d %s %s failed: %r", body.target, body.row_id, e)
        msg = str(e)[:500]
        _log_run(body, cost, False, time.time() - started, request_id, msg)
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
