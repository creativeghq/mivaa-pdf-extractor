"""Every live material image of a document ends linked to a product or excluded with a reason.

The printed label decides first; then a vision model picks among the series that page belongs to,
or says none; an image with no series to pick from is excluded. A failed vision call decides
nothing and leaves the image for the audit to count.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)

TASK = "shared_page_image_linking"
MAX_CANDIDATES = 20
MAX_IMAGES_PER_CALL = 16
PAGE_PX = 1600
KINDS = ("product", "decorative", "unclear")
EXCLUDED_BY = "settle_unlinked_images"
VECTOR_FLAGS = ("has_slig_embedding", "has_understanding_embedding", "has_color_slig",
                "has_texture_slig", "has_style_slig", "has_material_slig")
SCHEMA: Dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["images"],
    "properties": {"images": {"type": "array", "items": {
        "type": "object", "additionalProperties": False, "required": ["image", "series", "kind"],
        "properties": {"image": {"type": "integer"}, "series": {"type": ["integer", "null"]},
                       "kind": {"type": "string", "enum": list(KINDS)}}}}},
}


def page_candidates(page: int, printed: Set[str], product_of: Dict[str, str],
                    pages_of: Dict[str, Set[int]]) -> List[str]:
    """Product ids an image on `page` can belong to: series its half page prints, then series whose pages hold it."""
    out = [product_of[n] for n in sorted(printed) if n in product_of]
    out += [pid for pid, pages in sorted(pages_of.items()) if page in pages and pid not in out]
    return out[:MAX_CANDIDATES]


def verdicts(answer: Dict[str, Any], n_images: int, n_candidates: int) -> Dict[int, Tuple[str, Optional[int]]]:
    """{image index (0-based): (kind, candidate index or None)} for answers naming a real image.

    `product` with no valid candidate is `unclear`: a product image the model could not place is
    not attributable, and never a guess at the nearest series.
    """
    out: Dict[int, Tuple[str, Optional[int]]] = {}
    for a in answer.get("images") or []:
        i, s, kind = a.get("image"), a.get("series"), a.get("kind")
        if not isinstance(i, int) or not 1 <= i <= n_images or kind not in KINDS:
            continue
        if kind == "product" and isinstance(s, int) and 1 <= s <= n_candidates:
            out[i - 1] = ("product", s - 1)
        else:
            out[i - 1] = ("decorative" if kind == "decorative" else "unclear", None)
    return out


def exclusion_reason(kind: str) -> str:
    return "decorative_image" if kind == "decorative" else "unattributable_on_shared_page"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _render_half(pdf_bytes: bytes, pdf_idx: int, side: str) -> bytes:
    import fitz
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[pdf_idx]
        r = page.rect
        clip = fitz.Rect(r.x0, r.y0, r.x0 + r.width / 2, r.y1) if side == "left" else \
            fitz.Rect(r.x0 + r.width / 2, r.y0, r.x1, r.y1) if side == "right" else r
        zoom = PAGE_PX / max(clip.width, clip.height)
        return page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip).tobytes("png")
    finally:
        doc.close()


def _download(db: Any, row: Dict[str, Any]):
    from PIL import Image
    meta = row.get("metadata") or {}
    bucket, path = meta.get("storage_bucket"), meta.get("storage_path")
    if not bucket or not path:
        raise ValueError(f"image {row['id']} has no storage path")
    return Image.open(io.BytesIO(db.storage.from_(bucket).download(path)))


async def _ask(page_png: bytes, crops: Sequence[Any], names: Sequence[str], prompt: str, model: str,
               workspace_id: Optional[str]) -> Dict[int, Tuple[str, Optional[int]]]:
    from app.services.core.claude_tool_call import call_with_schema
    from app.services.images.gallery_coverage import numbered_sheet
    from app.services.utilities.prompt_registry import render
    listing = "\n".join(f"C{n + 1}: {name}" for n, name in enumerate(names))
    content = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(page_png).decode()}},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(numbered_sheet(crops, "I")).decode()}},
        {"type": "text", "text": render(prompt, candidates=listing, n_images=str(len(crops)))},
    ]
    res = await call_with_schema(task=TASK, model=model, max_tokens=3000, workspace_id=workspace_id,
                                 messages=[{"role": "user", "content": content}],
                                 output_config={"format": {"type": "json_schema", "schema": SCHEMA}})
    return verdicts(res.data, len(crops), len(names))


async def settle_unlinked_images(supabase: Any, document_id: str, pdf_bytes: bytes, apply: bool = True) -> Dict[str, Any]:
    """Link or exclude every unlinked live material or decorative image of the document; count each outcome."""
    from app.services.core.claude_models import SONNET
    from app.services.images.association_scores import PRINTED_LABEL_SCORE
    from app.services.images.label_linking import label_context, label_for, series_on, sole_series
    from app.services.pdf.ingestion_audit import owes_a_product
    from app.services.utilities.prompt_registry import load_prompt

    summary: Dict[str, Any] = {"unlinked": 0, "linked_by_label": 0, "linked_by_vision": 0, "excluded": defaultdict(int),
                               "undecided": 0, "failed_pages": []}
    ctx = await label_context(supabase, document_id, pdf_bytes)
    if ctx is None:
        return {**summary, "excluded": {}}
    db = ctx.db
    todo = [i for i in ctx.images if not ctx.links.get(i["id"]) and owes_a_product(i)]
    summary["unlinked"] = len(todo)
    if not todo:
        return {**summary, "excluded": {}}

    own = db.table("products").select("id, name, metadata").eq("source_document_id", document_id).execute().data or []
    pages_of = {p["id"]: {int(n) for n in ((p.get("metadata") or {}).get("page_range") or [])
                          if isinstance(n, (int, float)) and not isinstance(n, bool)} for p in own}
    name_of = {pid: name for name, pid in ctx.product_of.items()}
    name_of.update({p["id"]: p["name"] for p in own})

    def link(img: Dict[str, Any], product_id: str, method: str, **extra: Any) -> None:
        if apply:
            db.table("image_product_associations").upsert(
                {"workspace_id": img.get("workspace_id"), "product_id": product_id, "image_id": img["id"],
                 "overall_score": PRINTED_LABEL_SCORE, "metadata": {"association_method": method, **extra}},
                on_conflict="image_id,product_id").execute()

    async def exclude(img: Dict[str, Any], reason: str) -> None:
        summary["excluded"][reason] += 1
        if not apply:
            return
        from app.services.embeddings.vecs_service import get_vecs_service
        await get_vecs_service().delete_embeddings_by_image_ids([img["id"]])
        meta = dict(img.get("metadata") or {})
        meta["excluded_from_products"] = {"at": _now(), "reason": reason, "by": EXCLUDED_BY}
        db.table("document_images").update({"metadata": meta, **{f: False for f in VECTOR_FLAGS}}).eq("id", img["id"]).execute()

    halves: Dict[Tuple[int, str], List[Tuple[Dict[str, Any], List[str]]]] = defaultdict(list)
    for img in [i for i in ctx.images if not ctx.links.get(i["id"]) and i.get("image_type") == "DECORATIVE"
                and "excluded_from_products" not in (i.get("metadata") or {})]:
        await exclude(img, "decorative_image")
    for img in todo:
        half = ctx.lines_for(img["page_number"])
        lines = half[2] if half else []
        name = (label_for(img["bbox"], lines, ctx.series) if img.get("bbox") else sole_series(lines, ctx.series)) if lines else None
        if name and name in ctx.product_of:
            link(img, ctx.product_of[name], "printed_label", label=name)
            summary["linked_by_label"] += 1
            continue
        printed = set().union(*(series_on(l.tokens, ctx.series) for l in lines)) if lines else set()
        candidates = page_candidates(img["page_number"], printed, ctx.product_of, pages_of)
        if not candidates:
            await exclude(img, "no_series_on_page")
        elif half is None:
            summary["undecided"] += 1
        else:
            halves[(half[0], half[1])].append((img, candidates))

    if halves:
        prompt = await load_prompt("extraction", "shared_page_image_linking", stage="quality")
    for (pdf_idx, side), entries in halves.items():
        candidates = list(dict.fromkeys(pid for _, cands in entries for pid in cands))[:MAX_CANDIDATES]
        names = [name_of.get(pid, pid) for pid in candidates]
        try:
            page_png = await asyncio.to_thread(_render_half, pdf_bytes, pdf_idx, side)
            for start in range(0, len(entries), MAX_IMAGES_PER_CALL):
                batch = entries[start:start + MAX_IMAGES_PER_CALL]
                crops = [await asyncio.to_thread(_download, db, img) for img, _ in batch]
                answers = await _ask(page_png, crops, names, prompt, SONNET, batch[0][0].get("workspace_id"))
                for n, (img, _) in enumerate(batch):
                    if n not in answers:
                        summary["undecided"] += 1
                        continue
                    kind, pick = answers[n]
                    if kind == "product" and pick is not None:
                        link(img, candidates[pick], "shared_page_vision", series=names[pick])
                        summary["linked_by_vision"] += 1
                    else:
                        await exclude(img, exclusion_reason(kind))
        except Exception as e:
            logger.warning(f"settling unlinked images on pdf page {pdf_idx} ({side}) of {document_id} failed: {e}")
            summary["failed_pages"].append({"pdf_page": pdf_idx, "side": side, "error": f"{type(e).__name__}: {e}"[:200]})
            summary["undecided"] += len(entries)
    return {**summary, "excluded": dict(summary["excluded"])}

