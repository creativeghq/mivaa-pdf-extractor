"""Which colour of its series each product image shows, and whether it is a swatch or a room scene.

One vision call per product per sheet of up to SHEET_MAX images. The answer is constrained by a
per-product schema whose colour enum is that product's own colour list plus "multiple"/"unclear",
so a reply can only name a colour the product has. A filename tag (set by the database trigger) or
a manual tag is never overwritten.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

TASK = "image_variant_tagging"
CELL = 384
COLS = 4
SHEET_MAX = 12
SHOT_TYPES = ["swatch", "scene", "detail", "technical", "other"]
NOT_A_COLOUR = ["multiple", "unclear"]


def colour_list(metadata: Dict[str, Any]) -> List[Tuple[str, Optional[str]]]:
    """[(name, hex)] in catalogue order, from colour_options, else metadata.color."""
    import json

    out: List[Tuple[str, Optional[str]]] = []
    seen = set()
    for o in (metadata or {}).get("colour_options") or []:
        name = str((o or {}).get("name") or "").strip().lower()
        if name and name not in seen:
            seen.add(name)
            out.append((name, (o or {}).get("hex")))
    if out:
        return out
    raw = (metadata or {}).get("color")
    if isinstance(raw, str) and raw.strip().startswith("["):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = [raw]
    for c in raw if isinstance(raw, list) else ([raw] if isinstance(raw, str) else []):
        name = str(c or "").strip().lower()
        if name and name not in seen:
            seen.add(name)
            out.append((name, None))
    return out


def answer_schema(colours: Sequence[str]) -> Dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False, "required": ["images"],
        "properties": {"images": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["image", "shot_type", "colour"],
            "properties": {
                "image": {"type": "integer"},
                "shot_type": {"type": "string", "enum": SHOT_TYPES},
                "colour": {"type": "string", "enum": list(colours) + NOT_A_COLOUR},
            }}}},
    }


def numbered_sheet(images: Sequence[Image.Image]) -> bytes:
    rows = max(1, -(-len(images) // COLS))
    sheet = Image.new("RGB", (COLS * CELL, rows * CELL), "white")
    draw = ImageDraw.Draw(sheet)
    for n, img in enumerate(images):
        tile = img.convert("RGB").copy()
        tile.thumbnail((CELL - 8, CELL - 28))
        x, y = (n % COLS) * CELL, (n // COLS) * CELL
        sheet.paste(tile, (x + 4, y + 24))
        draw.text((x + 6, y + 4), f"I{n + 1}", fill=(200, 0, 0))
    out = io.BytesIO()
    sheet.save(out, "PNG")
    return out.getvalue()


def parse_answers(answer: Dict[str, Any], n: int, colours: Sequence[str]) -> Dict[int, Dict[str, str]]:
    """{0-based index: {shot_type, colour}} for answers naming a real photo and allowed values only."""
    allowed = set(colours) | set(NOT_A_COLOUR)
    out: Dict[int, Dict[str, str]] = {}
    for a in answer.get("images") or []:
        i = a.get("image")
        if isinstance(i, int) and 1 <= i <= n and a.get("shot_type") in SHOT_TYPES and a.get("colour") in allowed:
            out[i - 1] = {"shot_type": a["shot_type"], "colour": a["colour"]}
    return out


def association_update(colour: str) -> Dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    if colour in NOT_A_COLOUR:
        return {"variant_attributes": None, "variant_source": "vision", "variant_status": colour, "variant_tagged_at": now}
    return {"variant_attributes": {"color": colour}, "variant_source": "vision", "variant_status": "tagged", "variant_tagged_at": now}


def _pending_rows(db: Any, product_ids: Sequence[str], retag: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(product_ids), 50):
        chunk = list(product_ids[start:start + 50])
        page = 0
        while True:
            res = (db.table("image_product_associations")
                   .select("id, image_id, product_id, variant_source, variant_status, "
                           "document_images!inner(id, image_url, shot_type)")
                   .in_("product_id", chunk).order("id").range(page * 1000, page * 1000 + 999).execute())
            data = res.data or []
            rows.extend(data)
            if len(data) < 1000:
                break
            page += 1
    out = []
    for r in rows:
        img = r.get("document_images") or {}
        if not img.get("image_url"):
            continue
        needs_colour = r.get("variant_source") is None and (retag or r.get("variant_status") is None)
        needs_shot = img.get("shot_type") is None
        if needs_colour or needs_shot or (retag and r.get("variant_source") == "vision"):
            out.append(r)
    return out


async def _load_image(url: str) -> Optional[Image.Image]:
    from app.services.embeddings.understanding_backfill import _fetch_image_bytes

    body = await _fetch_image_bytes(url)
    if not body:
        return None
    try:
        return Image.open(io.BytesIO(body)).convert("RGB")
    except Exception:  # noqa: BLE001 - an unreadable file stays untagged and is reported
        return None


async def tag_product_images(
    *,
    product_ids: Optional[Sequence[str]] = None,
    document_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    job_id: Optional[str] = None,
    retag: bool = False,
    dry_run: bool = False,
    max_products: Optional[int] = None,
    log: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Tag shot type + colour on the images of the given products (or of one document's products)."""
    from app.services.core.claude_models import SONNET
    from app.services.core.claude_tool_call import ToolCallNotReturned, call_with_schema
    from app.services.core.supabase_client import get_supabase_client
    from app.services.utilities.prompt_registry import load_prompt, render

    log = log or logger
    db = get_supabase_client().client
    ids = list(product_ids or [])
    if document_id and not ids:
        img_ids = [r["id"] for r in (db.table("document_images").select("id").eq("document_id", document_id)
                                     .limit(5000).execute().data or [])]
        for start in range(0, len(img_ids), 200):
            res = (db.table("image_product_associations").select("product_id")
                   .in_("image_id", img_ids[start:start + 200]).execute())
            ids.extend(r["product_id"] for r in (res.data or []))
    if not ids and workspace_id:
        ids = [r["id"] for r in (db.table("products").select("id").eq("workspace_id", workspace_id)
                                 .limit(10000).execute().data or [])]
    ids = sorted(set(ids))

    products = {}
    for start in range(0, len(ids), 100):
        for p in db.table("products").select("id, name, workspace_id, metadata").in_("id", ids[start:start + 100]).execute().data or []:
            products[p["id"]] = p

    rows = await asyncio.to_thread(_pending_rows, db, list(products), retag)
    by_product: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_product.setdefault(r["product_id"], []).append(r)
    plan = sorted(by_product.items(), key=lambda kv: kv[0])
    if max_products:
        plan = plan[:max_products]
    summary: Dict[str, Any] = {
        "products": len(plan), "images": sum(len(v) for _, v in plan),
        "calls": sum(-(-len(v) // SHEET_MAX) for _, v in plan),
        "colour_tagged": 0, "multiple": 0, "unclear": 0, "shot_typed": 0,
        "unreadable": 0, "failed_calls": 0, "dry_run": dry_run,
    }
    if dry_run or not plan:
        return summary

    prompt = await load_prompt("extraction", "image_variant_tagging", stage="quality")
    shot_done: set = set()
    for pid, assoc in plan:
        product = products[pid]
        colours = colour_list(product.get("metadata") or {})
        names = [c for c, _ in colours]
        listing = "\n".join(f"- {c}: {h or 'no hex measured'}" for c, h in colours) or "- (no colour list: answer \"unclear\")"
        for start in range(0, len(assoc), SHEET_MAX):
            batch = assoc[start:start + SHEET_MAX]
            loaded = [(r, await _load_image(r["document_images"]["image_url"])) for r in batch]
            usable = [(r, im) for r, im in loaded if im is not None]
            summary["unreadable"] += len(loaded) - len(usable)
            if not usable:
                continue
            content = [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.b64encode(numbered_sheet([im for _, im in usable])).decode()}},
                {"type": "text", "text": render(prompt, product=product.get("name") or "", n_images=str(len(usable)), colours=listing)},
            ]
            try:
                res = await call_with_schema(
                    task=TASK, model=SONNET, max_tokens=2000, user_id=user_id, job_id=job_id,
                    workspace_id=product.get("workspace_id") or workspace_id, product_id=pid,
                    messages=[{"role": "user", "content": content}],
                    output_config={"format": {"type": "json_schema", "schema": answer_schema(names)}},
                )
            except ToolCallNotReturned as e:
                summary["failed_calls"] += 1
                log.warning(f"image_variant_tagging: no answer for product {pid}: {e}")
                continue
            for i, ans in parse_answers(res.data, len(usable), names).items():
                row = usable[i][0]
                image_id = row["document_images"]["id"]
                if row["document_images"].get("shot_type") is None and image_id not in shot_done:
                    db.table("document_images").update({"shot_type": ans["shot_type"], "shot_type_source": "vision"}) \
                        .eq("id", image_id).is_("shot_type", "null").execute()
                    shot_done.add(image_id)
                    summary["shot_typed"] += 1
                if not names or row.get("variant_source") in ("filename", "manual"):
                    continue
                db.table("image_product_associations").update(association_update(ans["colour"])).eq("id", row["id"]).execute()
                key = "colour_tagged" if ans["colour"] not in NOT_A_COLOUR else ans["colour"]
                summary[key] += 1
    log.info(f"🎨 Image variant tagging: {summary}")
    return summary
