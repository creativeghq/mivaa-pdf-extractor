"""One row per product with its first images, so an ingestion can be looked at in a minute."""

from __future__ import annotations

import asyncio
import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from PIL import Image, ImageDraw, ImageFont

THUMBS_PER_ROW = 6
THUMB = 160
LABEL_W = 240
ROWS_PER_SHEET = 20
PAD = 8


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _thumb(data: Optional[bytes]) -> Image.Image:
    tile = Image.new("RGB", (THUMB, THUMB), (235, 235, 235))
    if not data:
        return tile
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return tile
    native = max(img.size)
    img.thumbnail((THUMB, THUMB), Image.LANCZOS)
    tile.paste(img, ((THUMB - img.width) // 2, (THUMB - img.height) // 2))
    write = ImageDraw.Draw(tile).text
    write((4, THUMB - 16), f"{native}px", fill=(200, 0, 0), font=_font(12))
    return tile


def render_sheet(rows: List[Dict[str, Any]]) -> bytes:
    """rows: {"name", "count", "images": [bytes|None]}; the native size is printed on each thumbnail."""
    width = LABEL_W + THUMBS_PER_ROW * (THUMB + PAD) + PAD
    sheet = Image.new("RGB", (width, len(rows) * (THUMB + PAD) + PAD), "white")
    write = ImageDraw.Draw(sheet).text
    for n, row in enumerate(rows):
        y = PAD + n * (THUMB + PAD)
        write((PAD, y + 4), row["name"][:30], fill="black", font=_font(16))
        write((PAD, y + 26), f"{row['count']} image(s)", fill=(90, 90, 90), font=_font(13))
        for k, data in enumerate(row["images"][:THUMBS_PER_ROW]):
            sheet.paste(_thumb(data), (LABEL_W + k * (THUMB + PAD), y))
    out = io.BytesIO()
    sheet.save(out, "JPEG", quality=80)
    return out.getvalue()


async def build_contact_sheets(supabase: Any, document_id: str) -> Dict[str, Any]:
    db = getattr(supabase, "client", supabase)
    products = db.table("products").select("id, name").eq("source_document_id", document_id).order("name").execute().data or []
    rows: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=30) as http:
        for p in products:
            links = db.table("image_product_associations").select("overall_score, document_images!inner(image_url, metadata)") \
                .eq("product_id", p["id"]).execute().data or []
            live = sorted((l for l in links if "excluded_from_products" not in (l["document_images"].get("metadata") or {})),
                          key=lambda l: -(l.get("overall_score") or 0))
            urls = [l["document_images"]["image_url"] for l in live[:THUMBS_PER_ROW] if l["document_images"].get("image_url")]

            async def get(u: str) -> Optional[bytes]:
                try:
                    r = await http.get(u)
                    return r.content if r.status_code == 200 else None
                except Exception:
                    return None

            rows.append({"name": p["name"], "count": len(live), "images": await asyncio.gather(*(get(u) for u in urls))})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sheets = []
    for n in range(0, len(rows), ROWS_PER_SHEET):
        path = f"extracted/{document_id}/contact_sheet/{stamp}_{n // ROWS_PER_SHEET + 1}.jpg"
        data = await asyncio.to_thread(render_sheet, rows[n:n + ROWS_PER_SHEET])
        db.storage.from_("pdf-tiles").upload(path, data, {"content-type": "image/jpeg"})
        sheets.append(db.storage.from_("pdf-tiles").get_public_url(path).rstrip("?"))
    return {"sheets": sheets, "products": len(rows), "without_images": [r["name"] for r in rows if not r["count"]]}
