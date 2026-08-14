"""
Product spec extraction via Claude Vision on rendered PDF spec pages.

Why Claude Vision instead of per-icon OCR:
  Ceramic catalog icon strips use stylized vector glyphs that document OCR
  (PaddleOCR) struggles with at icon scale. Individual 67x67 px icon crops also
  fail because they lack surrounding context — no label, no value, nothing
  to anchor the interpretation.

  Rendering the full PDF page at 300 DPI and passing it to Claude Haiku Vision
  gives Claude both the icon AND the surrounding text/layout. We consistently
  recover: product_name, dimensions, body_type, colors, variants/SKUs,
  pieces_per_box, m²/box, sqft/box, weight, pallet info, and (for pages with
  per-product spec grids) slip/PEI/fire/shade/frost ratings.

Cost: ~1 Claude Haiku Vision call per product spec page (~2000 input tokens,
~600 output tokens) = ~$0.001-0.002 per product. Cheap.

Runs after Stage 4.7 (chunk+vision_analysis rollup) and only fills fields that
are still null/empty. Never overwrites AI values that already exist.
"""

import base64
import io
import json
import logging
import os
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from PIL import Image

from app.services.core.anthropic_error_reporter import report_anthropic_failure
from app.services.utilities.prompt_registry import get_cached

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Haiku 4.5 is plenty capable for ceramic tile spec extraction and 10x cheaper
# than Opus. If results are weak on a specific catalog, we can override via
# env var without changing code.
CLAUDE_VISION_MODEL = os.getenv("PRODUCT_SPEC_VISION_MODEL", "claude-haiku-4-5")

# Render DPI for PDF pages. 280 is high enough for the technical-characteristics
# icon strip (slip resistance / PEI / water absorption glyphs are ~20-30 px tall
# at 220 DPI and Claude Haiku struggles to classify them; at 280 they're
# clearly readable). When the resulting PNG exceeds 5 MB, _render_page_under_limit
# drops DPI step-by-step and switches to JPEG so we always fit within Claude's
# hard limit without losing the icons.
PAGE_RENDER_DPI = 280

# Max bytes per image before we downscale. 4 MB leaves comfortable headroom
# under Claude's 5 MB hard limit after base64 expansion overhead.
MAX_IMAGE_BYTES = 4_000_000

# Prompt: extraction / image_analysis / product_spec_vision (#347 phase 3P).


def _build_spec_prompt(product_name: str) -> str:
    """Fill the stored spec template with the target product name.

    We use a plain `.replace` rather than `.format` because the template
    embeds literal JSON braces that would trip .format's {…} parser.
    """
    safe = (product_name or "").strip() or "the ceramic product on this page"
    template = get_cached("extraction", "product_spec_vision", stage="image_analysis")
    return template.replace("{product_name}", safe)


# Was `SPEC_PROMPT = _build_spec_prompt(...)`, evaluated at IMPORT time — which can neither
# await nor reach the prompt store. A function instead, resolved when it is actually needed
# (#347 phase 3P).
def default_spec_prompt() -> str:
    return _build_spec_prompt("the ceramic product on this page")


def _render_pdf_page_to_bytes(
    pdf_path: str,
    page_index: int,
    dpi: int = PAGE_RENDER_DPI,
    *,
    fmt: str = "png",
    jpeg_quality: int = 85,
) -> bytes:
    """Render one PDF page to image bytes.

    `fmt` may be "png" or "jpg". JPEG is ~4-6x smaller for photographic
    brochure pages and is used as a fallback when PNG renders exceed the
    5 MB Claude limit (some high-res Harmony spreads produce 6-8 MB PNGs).
    """
    doc = fitz.open(pdf_path)
    try:
        pix = doc[page_index].get_pixmap(dpi=dpi)
        if fmt == "jpg":
            return pix.tobytes("jpg", jpg_quality=jpeg_quality)
        return pix.tobytes("png")
    finally:
        doc.close()


def _render_page_under_limit(
    pdf_path: str,
    page_index: int,
    max_bytes: int = MAX_IMAGE_BYTES,
) -> Optional[bytes]:
    """Render a PDF page into bytes guaranteed to fit under `max_bytes`.

    Strategy:
      1. Try PNG at the default DPI. Keep if it's already under max_bytes.
      2. If too big, try PNG at progressively lower DPIs (180, 150, 120).
      3. If PNG still too big, switch to JPEG at 180/150/120 DPI with
         quality 88/82/75.
      4. Return the smallest rendering we can produce; return None only
         if every attempt fails at the PyMuPDF level.

    This bypasses PIL entirely — PyMuPDF produces both PNG and JPEG
    natively, so we never have to round-trip through Image.open, which
    was choking on high-res Harmony pages with "cannot identify image
    file".
    """
    # Pass 1: PNG at several DPIs
    for dpi in (PAGE_RENDER_DPI, 180, 150, 120):
        try:
            data = _render_pdf_page_to_bytes(pdf_path, page_index, dpi=dpi, fmt="png")
        except Exception as e:
            logger.warning(f"   ⚠️ PNG render page {page_index} @ {dpi} dpi failed: {e}")
            continue
        if len(data) <= max_bytes:
            return data

    # Pass 2: JPEG at several DPI / quality combos
    for dpi, q in ((180, 88), (150, 85), (120, 80), (100, 75)):
        try:
            data = _render_pdf_page_to_bytes(
                pdf_path, page_index, dpi=dpi, fmt="jpg", jpeg_quality=q,
            )
        except Exception as e:
            logger.warning(f"   ⚠️ JPEG render page {page_index} @ {dpi} dpi q={q} failed: {e}")
            continue
        if len(data) <= max_bytes:
            return data

    # Last-resort attempt — even an oversized payload is preferable to None
    # since the caller logs the failure and moves on.
    try:
        return _render_pdf_page_to_bytes(pdf_path, page_index, dpi=100, fmt="jpg", jpeg_quality=70)
    except Exception:
        return None


def _shrink_if_needed(png_bytes: bytes, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    """Downscale a PNG (and optionally flatten to JPEG) to fit under max_bytes.

    Guarantees a return value under max_bytes whenever possible. Tries PNG at
    progressively smaller sizes first; if PNG can't get small enough (very
    high-res brochure pages with many gradients compress poorly), falls back
    to JPEG at quality 85 which is ~4-6x more efficient for photographic
    catalog content. Icon glyphs and spec table text survive JPEG 85.

    PIL occasionally fails to open a valid-looking PNG produced by PyMuPDF
    (seen on high-resolution Harmony spread pages — raises "cannot identify
    image file"). In that case we return the original bytes unchanged and
    let the caller either send them through to Claude (if under 5 MB) or
    log a downstream failure. Never propagate the PIL error — a single
    unreadable page should not abort the rest of the scan.
    """
    if len(png_bytes) <= max_bytes:
        return png_bytes
    try:
        im = Image.open(io.BytesIO(png_bytes))
        im.load()  # force decode so errors surface here, not later
    except Exception as e:
        logger.warning(
            f"product_spec_vision_extractor: PIL could not open {len(png_bytes)//1024} KB "
            f"PNG from PyMuPDF ({e}); returning raw bytes unchanged"
        )
        return png_bytes
    # Flatten alpha to white so JPEG fallback works if we need it.
    if im.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1])
        im = bg
    elif im.mode != "RGB":
        im = im.convert("RGB")

    # Pass 1: PNG at descending sizes.
    last_png = png_bytes
    for edge in (2200, 1800, 1400, 1100, 900, 700):
        scaled = im.copy()
        scaled.thumbnail((edge, edge))
        buf = io.BytesIO()
        scaled.save(buf, format="PNG", optimize=True)
        out = buf.getvalue()
        last_png = out
        if len(out) <= max_bytes:
            return out

    # Pass 2: JPEG fallback — always fits thanks to much better compression.
    for edge, quality in ((2200, 88), (1800, 88), (1400, 85), (1100, 82)):
        scaled = im.copy()
        scaled.thumbnail((edge, edge))
        buf = io.BytesIO()
        scaled.save(buf, format="JPEG", quality=quality, optimize=True)
        out = buf.getvalue()
        if len(out) <= max_bytes:
            return out

    # Last resort: return whatever is smallest. _call_claude_vision will still
    # attempt it; a 400 from Claude is preferable to silently dropping a page.
    return last_png


def _detect_image_media_type(image_bytes: bytes) -> str:
    """Sniff PNG vs JPEG from the first few bytes so we can tell Claude which
    media_type to use after _shrink_if_needed may have re-encoded to JPEG."""
    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return "image/png"  # fallback


def _call_claude_vision(
    png_bytes: bytes,
    prompt: Optional[str] = None,
    *,
    job_id: Optional[str] = None,
    product_id: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Single Claude Vision call, returns parsed JSON dict or None on failure.

    `job_id` + `product_id` are forwarded to `tracked_claude_call` so per-
    product cost attribution lands in `ai_usage_logs.product_id` (audit
    fix: previously every Stage 4.7 spec-vision call landed without
    product_id and showed up as orphan spend on the cost dashboard).

    `model` overrides the module-default `CLAUDE_VISION_MODEL` for this call
    ONLY (Tier B passes Opus). Passing it per-call — instead of mutating the
    module global — is what keeps concurrent jobs from leaking each other's
    model choice (the S4-1 race).
    """
    if prompt is None:
        prompt = default_spec_prompt()
    if not ANTHROPIC_API_KEY:
        logger.error("product_spec_vision_extractor: ANTHROPIC_API_KEY not set")
        return None

    image_bytes = _shrink_if_needed(png_bytes)
    media_type = _detect_image_media_type(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    logger.info(
        f"   📤 spec vision: sending {len(image_bytes)//1024} KB {media_type} to Claude"
    )

    try:
        from app.services.core.claude_helper import tracked_claude_call
        resp = tracked_claude_call(
            task="product_spec_vision_extraction",
            model=model or CLAUDE_VISION_MODEL,
            max_tokens=3000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
            job_id=job_id,
            product_id=product_id,
        )
    except Exception as e:
        report_anthropic_failure(e, service="product_spec_vision_extractor")
        logger.warning(f"product_spec_vision_extractor: Claude call failed: {e}")
        return None

    text = resp.content[0].text if resp.content else ""
    stripped = text.strip()
    # Strip markdown fences if Claude ignored the "no fences" instruction
    if stripped.startswith("```"):
        inner = stripped.split("```", 2)
        if len(inner) >= 2:
            stripped = inner[1]
            if stripped.startswith("json"):
                stripped = stripped[4:]
            stripped = stripped.strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError as e:
        logger.warning(
            f"product_spec_vision_extractor: JSON parse failed ({e}); "
            f"raw[:200]={stripped[:200]!r}"
        )
        return None


def _get_source_pdf_path(document_id: str) -> Optional[str]:
    """Find the source PDF for a document on disk.

    PDFs are kept under /tmp/pdf_processor_{document_id}/{document_id}.pdf during
    processing. If the temp dir was cleaned, we can re-download from Supabase
    storage via the documents.file_path field — handled by the caller.

    Also rejects 0-byte files: when the previous orchestrator crashed mid-
    write (e.g. kernel OOM kill on Apr 29), the temp dir was left with an
    empty file. Treating it as "found" then handing it to pymupdf produces
    the "Cannot open empty file" error chain. Returning None instead lets
    the caller re-download from Supabase storage.
    """
    candidates = [
        f"/tmp/pdf_processor_{document_id}/{document_id}.pdf",
        f"/tmp/pdf_processor_{document_id}/source.pdf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            if size > 0:
                return p
            # Empty file — clean it up so the caller's re-download writes
            # cleanly without an EEXIST or partial-write conflict.
            try:
                os.unlink(p)
            except OSError:
                pass
    return None


# Per-document memo for the layout-cache text map. The cache is immutable for the
# life of a job, so reading it once per document (not once per product) is safe.
# Bounded so a long-lived worker doesn't accumulate full-document text maps.
_CACHE_TEXT_BY_DOC: Dict[str, Dict[int, str]] = {}
_CACHE_TEXT_MAX_DOCS = 4


def _normalize_for_match(s: str) -> str:
    """Accent-strip + uppercase so 'PIQUÉ' matches 'PIQUE' for name lookup."""
    import unicodedata
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.upper().strip()


def _load_cache_page_texts(document_id: Optional[str]) -> Dict[int, str]:
    """0-indexed PDF page → PaddleOCR-VL reading-order text, from the layout cache.

    Lets name-based page resolution match products whose name is rendered INSIDE
    a page image (designer fonts / logos / stylized titles) — the embedded PDF
    text layer misses those, but the VLM OCR'd them into
    ``document_layout_analysis``. Returns ``{}`` on any failure / no document_id,
    so the caller falls back to the raw text layer only (never raises).

    Memoized per document (see :data:`_CACHE_TEXT_BY_DOC`) so a catalog with many
    image-baked-name products reads the layout table ONCE, not once per product.
    """
    if not document_id:
        return {}
    memo = _CACHE_TEXT_BY_DOC.get(document_id)
    if memo is not None:
        return memo
    try:
        from app.services.core.supabase_client import get_supabase_client
        from app.api.pdf_processing.stage_1_layout_precompute import (
            load_page_texts_from_cache,
        )
        # 0-indexed to match PDF page indices used by doc[idx] in the name scan.
        result = load_page_texts_from_cache(
            get_supabase_client(), document_id, logger=logger, zero_indexed=True,
        )
    except Exception as e:
        logger.debug(f"   spec vision: layout cache read failed (non-fatal): {e}")
        return {}
    # Memoize only non-empty successes, so a transient failure / no-rows can retry
    # on the next product instead of caching an empty map. Evict oldest on overflow.
    if result:
        if len(_CACHE_TEXT_BY_DOC) >= _CACHE_TEXT_MAX_DOCS:
            oldest = next(iter(_CACHE_TEXT_BY_DOC), None)
            if oldest is not None:
                _CACHE_TEXT_BY_DOC.pop(oldest, None)  # crash-safe under concurrency
        _CACHE_TEXT_BY_DOC[document_id] = result
    return result


def _find_pages_by_name_in_texts(
    page_texts: Dict[int, str], product_name: str, max_pages: int = 12,
) -> List[int]:
    """0-indexed pages whose (normalized) reading-order text contains the name.

    The cache-backed counterpart to :func:`_find_pdf_pages_by_text` — used as the
    fallback that locates products whose name is baked into a page image, without
    re-opening the PDF.
    """
    needle = _normalize_for_match(product_name)
    if not needle:
        return []
    out: List[int] = []
    for i in sorted(page_texts):
        if needle in _normalize_for_match(page_texts[i]):
            out.append(i)
            if len(out) >= max_pages:
                break
    return out


def _find_pdf_pages_by_text(
    pdf_path: str,
    product_name: str,
    max_pages: int = 12,
) -> List[int]:
    """Scan the PDF TEXT LAYER and return 0-indexed page indices whose text
    contains `product_name` (case- and accent-insensitive).

    This is the authoritative signal for "where does this product live in the
    PDF" when the chunk metadata's `product_pages` turns out to be catalog
    folio labels (two per physical spread) rather than absolute PDF indices.
    Pages whose name is baked into an image (empty text layer) are handled by
    the cache-backed :func:`_find_pages_by_name_in_texts` fallback in the resolver.
    """
    needle = _normalize_for_match(product_name)
    if not needle:
        return []

    doc = fitz.open(pdf_path)
    matches: List[int] = []
    try:
        for i in range(doc.page_count):
            if needle in _normalize_for_match(doc[i].get_text()):
                matches.append(i)
                if len(matches) >= max_pages:
                    break
    finally:
        doc.close()
    return matches


def _resolve_pdf_pages_for_product(
    pdf_path: str,
    product_page_range: List[int],
    product_name: Optional[str] = None,
    document_id: Optional[str] = None,
) -> List[int]:
    """Return 0-indexed PDF page indices where `product_name` actually lives.

    Priority:
      1. **Text scan** (authoritative): open the PDF and find every page
         that literally contains the product name. Accent/case-insensitive.
      2. **Fallback**: treat `product_page_range` as 1-indexed PDF page
         numbers and subtract 1. This is only correct when the upstream
         pipeline stored true PDF page numbers (which it does for some
         catalogs, but NOT for Harmony-style catalogs where the chunk
         metadata stores printed folio labels — two folios per spread).

    Background: earlier revisions trusted `product_page_range` and applied a
    fuzzy `(-2..+1)` offset heuristic, and then a strict `n - 1` conversion.
    Both were wrong for Harmony: chunk metadata stored catalog folio labels
    like [26, 27, 28, 29, 30, 31] for a product whose actual PDF pages were
    [13, 14, 15] (one physical page = two printed folios). Claude Vision was
    scanning brand-intro pages that had no VALENOVA data on them at all.

    We still honor `product_page_range` as a fallback so products with no
    name match (e.g. renamed, SKU-only) can still get scanned.
    """
    if not pdf_path or not os.path.exists(pdf_path):
        return []

    doc = fitz.open(pdf_path)
    total = doc.page_count
    doc.close()

    # Primary: name-based text scan. Raw PDF text layer FIRST (in-memory, no DB);
    # only on a miss do we pay the layout-cache read — the image-baked-name
    # minority — so born-digital catalogs never touch the layout table here, and
    # the cache read is memoized per document so even an all-image catalog reads
    # it once rather than per product.
    text_matches: List[int] = []
    if product_name:
        text_matches = _find_pdf_pages_by_text(pdf_path, product_name)
        if not text_matches and document_id:
            text_matches = _find_pages_by_name_in_texts(
                _load_cache_page_texts(document_id), product_name,
            )

    # Fallback: numeric conversion from chunk metadata
    numeric_matches: List[int] = sorted({
        int(p) - 1
        for p in (product_page_range or [])
        if isinstance(p, (int, str)) and str(p).isdigit() and 0 <= int(p) - 1 < total
    })

    if text_matches:
        logger.info(
            f"   🗺  spec vision: text scan found '{product_name}' on PDF pages "
            f"{text_matches} (total={total}) — using these"
        )
        return text_matches

    if numeric_matches:
        logger.info(
            f"   🗺  spec vision: text scan empty, falling back to numeric "
            f"input={sorted(product_page_range)} → 0-indexed {numeric_matches} "
            f"(total={total})"
        )
        return numeric_matches

    logger.info(
        f"   🗺  spec vision: no pages resolvable for '{product_name}' "
        f"(input={sorted(product_page_range) if product_page_range else []})"
    )
    return []


def _select_best_spec_result(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge multiple page extractions, preferring the most populated.

    Strategy: field-by-field, keep the first non-null/non-empty value seen
    across all pages. This handles catalogs where different fields live on
    different spec pages (e.g. packing on one, icons on another).
    """
    merged: Dict[str, Any] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        for key, value in result.items():
            # Don't overwrite a real value with null/empty
            existing = merged.get(key)
            if existing in (None, [], "") and value not in (None, [], ""):
                merged[key] = value
            # For lists, merge uniquely
            elif isinstance(existing, list) and isinstance(value, list):
                seen = {repr(x) for x in existing}
                for v in value:
                    if repr(v) not in seen:
                        existing.append(v)
                        seen.add(repr(v))
    return merged


def extract_specs_from_pdf_pages(
    pdf_path: str,
    product_page_range: List[int],
    product_name: Optional[str] = None,
    *,
    job_id: Optional[str] = None,
    product_id: Optional[str] = None,
    document_id: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract ceramic-tile specifications from a product's PDF spec pages.

    Args:
        pdf_path: Path to the source PDF on disk.
        product_page_range: 1-indexed PDF page numbers for this product.
        product_name: REQUIRED in practice — embedded in the prompt so Claude
                      filters multi-product pages down to this product's rows
                      only. Pages whose Claude response has
                      `page_contains_target=false` are dropped entirely.

    Returns:
        Merged spec dict (schema defined by the extraction/product_spec_vision prompt).
        Empty on failure.
    """
    if not os.path.exists(pdf_path):
        logger.warning(f"product_spec_vision_extractor: PDF not found at {pdf_path}")
        return {}

    pdf_indices = _resolve_pdf_pages_for_product(
        pdf_path, product_page_range, product_name=product_name,
        document_id=document_id,
    )
    if not pdf_indices:
        logger.info(f"product_spec_vision_extractor: no valid pages for {product_name}")
        return {}

    pdf_indices = pdf_indices[:8]

    product_aware_prompt = _build_spec_prompt(product_name or "")
    logger.info(
        f"📸 product_spec_vision_extractor: {product_name or '?'} "
        f"scanning {len(pdf_indices)} pages {pdf_indices}"
    )

    results: List[Dict[str, Any]] = []
    pages_kept = 0
    pages_dropped_other_product = 0

    for idx in pdf_indices:
        # Render directly under the 5 MB Claude limit via PyMuPDF (PNG then
        # JPEG fallback at progressively lower DPI). This bypasses PIL
        # entirely, which previously choked on some high-res Harmony
        # spreads with "cannot identify image file".
        image_bytes = _render_page_under_limit(pdf_path, idx)
        if image_bytes is None:
            logger.warning(f"   ⚠️ render page {idx} failed at every DPI/format attempt")
            continue

        try:
            data = _call_claude_vision(
                image_bytes,
                prompt=product_aware_prompt,
                job_id=job_id,
                product_id=product_id,
                model=model,
            )
        except Exception as e:
            logger.warning(f"   ⚠️ Claude Vision call failed for page {idx}: {e}")
            continue

        if not data:
            continue

        # Skip pages that explicitly said they don't contain target product data.
        if data.get("page_contains_target") is False:
            pages_dropped_other_product += 1
            logger.info(
                f"   ⏭  page {idx}: Claude reported no '{product_name}' data, skipping"
            )
            continue

        results.append(data)
        pages_kept += 1

        # Early break: once we have both the icon strip values AND the packing
        # block for the target product, scanning further pages only adds noise.
        has_icons = any(
            data.get(k) not in (None, [], "")
            for k in ("slip_resistance", "pei_rating", "water_absorption_class",
                      "shade_variation")
        )
        has_packing = all(
            data.get(k) not in (None, [], "")
            for k in ("pieces_per_box", "m2_per_box", "weight_per_box_kg", "boxes_per_pallet")
        )
        if has_icons and has_packing:
            logger.info(
                f"   ✅ page {idx} has full icon strip + packing for '{product_name}', "
                f"stopping scan"
            )
            break

    if not results:
        logger.info(
            f"   ℹ️ product_spec_vision_extractor: no pages matched '{product_name}' "
            f"({pages_dropped_other_product} pages belonged to other products)"
        )
        return {}

    merged = _select_best_spec_result(results)
    # Strip out the envelope flags before returning — callers don't need them.
    merged.pop("page_contains_target", None)

    populated = sum(1 for v in merged.values() if v not in (None, [], ""))
    logger.info(
        f"   ✅ product_spec_vision_extractor: {populated}/{len(merged)} fields populated "
        f"from {pages_kept} page(s) ({pages_dropped_other_product} other-product pages dropped)"
    )
    return merged


def map_vision_specs_to_product_metadata(
    specs: Dict[str, Any],
) -> Dict[str, Any]:
    """Transform the flat vision result into the nested product.metadata shape.

    The vision extractor returns a flat dict like {slip_resistance: "R9", ...}
    but products.metadata is nested: {performance: {slip_resistance: "R9"}, ...}.
    This function performs that mapping.

    Returns a dict with the same nested shape used by Stage 4.7's
    _merge_enriched_fields_into_metadata — so the caller can feed it directly
    into the existing merge pipeline.
    """
    out: Dict[str, Any] = {}

    # Material properties
    mp = {}
    if specs.get("finish"):           mp["finish"] = specs["finish"]
    if specs.get("body_type"):        mp["body_type"] = specs["body_type"]
    if specs.get("thickness_mm") is not None:
        mp["thickness_mm"] = specs["thickness_mm"]
    if specs.get("patterns"):         mp["patterns"] = specs["patterns"]
    if mp:
        out["material_properties"] = mp

    # Performance (the spec icon block)
    perf = {}
    for k in ("slip_resistance", "pei_rating", "water_absorption_class",
              "water_absorption_pct", "fire_rating", "frost_resistance",
              "shade_variation", "traffic_level"):
        if specs.get(k) not in (None, [], ""):
            perf[k] = specs[k]
    if perf:
        out["performance"] = perf

    # Application
    app = {}
    if specs.get("recommended_use"):       app["recommended_use"] = specs["recommended_use"]
    if specs.get("installation_method"):   app["installation_method"] = specs["installation_method"]
    if specs.get("joint_width_mm") is not None:
        app["joint_width_mm"] = specs["joint_width_mm"]
    if app:
        out["application"] = app

    # Packaging — scalar (per product) fields
    pkg = {}
    for k in ("pieces_per_box", "m2_per_box", "sqft_per_box",
              "weight_per_box_kg", "weight_per_box_lb",
              "boxes_per_pallet", "m2_per_pallet",
              "weight_per_pallet_kg", "weight_per_pallet_lb"):
        if specs.get(k) is not None:
            pkg[k] = specs[k]
    # Per-variant packaging rows (when the catalog's packing table lists
    # different pcs/box or weight per format). UI renders as a table.
    per_variant = specs.get("packaging_per_variant")
    if isinstance(per_variant, list) and per_variant:
        cleaned = [row for row in per_variant if isinstance(row, dict) and any(
            row.get(k) not in (None, "", []) for k in row.keys()
        )]
        if cleaned:
            pkg["per_variant"] = cleaned
    if pkg:
        out["packaging"] = pkg

    # Commercial — grout recommendations have their own shape
    commercial = {}
    if specs.get("grout_recommendations"):
        commercial["grout_details"] = specs["grout_recommendations"]
    if specs.get("variants"):
        commercial["vision_variants"] = specs["variants"]
    if commercial:
        out["commercial"] = commercial

    # Compliance
    compl = {}
    if specs.get("certifications"):
        compl["certifications"] = specs["certifications"]
    if compl:
        out["compliance"] = compl

    # Dimensions — if vision got them, promote to top-level (pair with existing)
    if specs.get("dimensions_cm"):
        out["dimensions_cm_from_vision"] = specs["dimensions_cm"]
    if specs.get("dimensions_inch"):
        out["dimensions_inch_from_vision"] = specs["dimensions_inch"]

    return out
