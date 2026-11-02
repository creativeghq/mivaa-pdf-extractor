"""Does the maker's gallery already show what a small catalogue image shows?

The pixel matcher cannot compare a square swatch with a landscape room photo, so this asks a vision
model, once per product: a numbered sheet of the gallery, then each small image. Only answers that
name a real image and a real gallery photo count; anything else is "not covered", so an image is
never retired on a guess.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

TASK = "gallery_coverage"
CELL = 320
GALLERY_MAX = 12
SMALL_MAX = 16
SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["images"],
    "properties": {"images": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["image", "covered", "gallery_photo"],
        "properties": {"image": {"type": "integer"}, "covered": {"type": "boolean"},
                       "gallery_photo": {"type": ["integer", "null"]}}}}},
}


def numbered_sheet(images: Sequence[Image.Image], prefix: str, cols: int = 4) -> bytes:
    """A grid of images, each labelled `<prefix><n>` from 1, as PNG."""
    rows = max(1, -(-len(images) // cols))
    sheet = Image.new("RGB", (cols * CELL, rows * CELL), "white")
    write = ImageDraw.Draw(sheet).text
    for n, img in enumerate(images):
        tile = img.convert("RGB").copy()
        tile.thumbnail((CELL - 8, CELL - 28))
        x, y = (n % cols) * CELL, (n // cols) * CELL
        sheet.paste(tile, (x + 4, y + 24))
        write((x + 6, y + 4), f"{prefix}{n + 1}", fill=(200, 0, 0))
    out = io.BytesIO()
    sheet.save(out, "PNG")
    return out.getvalue()


def covered_indexes(answer: Dict[str, Any], n_small: int, n_gallery: int) -> Dict[int, int]:
    """{small image index (0-based): gallery photo index (0-based)} for answers that name real images."""
    out: Dict[int, int] = {}
    for a in answer.get("images") or []:
        i, g = a.get("image"), a.get("gallery_photo")
        if a.get("covered") is True and isinstance(i, int) and isinstance(g, int) and 1 <= i <= n_small and 1 <= g <= n_gallery:
            out[i - 1] = g - 1
    return out


async def coverage(small: Sequence[Image.Image], gallery: Sequence[Image.Image], *, product_name: str, model: str,
                   prompt: str, user_id: Optional[str] = None, workspace_id: Optional[str] = None,
                   product_id: Optional[str] = None) -> Dict[int, int]:
    """Map each covered small image (index into `small`) to the gallery photo that shows it."""
    from app.services.core.claude_tool_call import call_with_schema
    from app.services.utilities.prompt_registry import render
    gallery = list(gallery)[:GALLERY_MAX]
    result: Dict[int, int] = {}
    for start in range(0, len(small), SMALL_MAX):
        batch = list(small)[start:start + SMALL_MAX]
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(numbered_sheet(gallery, "G")).decode()}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(numbered_sheet(batch, "S")).decode()}},
            {"type": "text", "text": render(prompt, product=product_name, n_gallery=str(len(gallery)), n_small=str(len(batch)))},
        ]
        res = await call_with_schema(task=TASK, model=model, max_tokens=1500, user_id=user_id, workspace_id=workspace_id,
                                     product_id=product_id, messages=[{"role": "user", "content": content}],
                                     output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
        for i, g in covered_indexes(res.data, len(batch), len(gallery)).items():
            result[start + i] = g
    return result


def split_covered(ids: Sequence[str], covered: Dict[int, int], gallery_ids: Sequence[str]) -> Tuple[Dict[str, str], List[str]]:
    """({small id: covering gallery id}, [small ids not covered])."""
    hit = {ids[i]: gallery_ids[g] for i, g in covered.items() if i < len(ids) and g < len(gallery_ids)}
    return hit, [i for i in ids if i not in hit]
