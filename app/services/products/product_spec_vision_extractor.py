"""
Product spec extraction via Claude Vision on rendered PDF spec pages.

Why Claude Vision instead of per-icon OCR:
  Ceramic catalog icon strips use stylized vector glyphs that neither Chandra
  (document OCR) nor EasyOCR (word-level OCR) can read. Individual 67x67 px
  icon crops also fail because they lack surrounding context — no label, no
  value, nothing to anchor the interpretation.

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
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic
import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Haiku 4.5 is plenty capable for ceramic tile spec extraction and 10x cheaper
# than Sonnet. If results are weak on a specific catalog, we can override via
# env var without changing code.
CLAUDE_VISION_MODEL = os.getenv("PRODUCT_SPEC_VISION_MODEL", "claude-haiku-4-5-20251001")

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

# The single prompt we send for every product spec page. The schema is
# deliberately flat — nested objects make the post-parse merge harder.
# Placeholder {product_name} is substituted at call time so Claude can
# filter multi-product spec pages down to just the one we're enriching.
SPEC_PROMPT_TEMPLATE = """You are reading one page of a ceramic tile catalog. The page may list MULTIPLE product series (e.g. "PIQUE 30", "PIQUÉ WAFFLE", "VALENOVA", "ONA", "FOLD") side by side in a single packing table or shared spec grid. We are enriching this specific product:

    TARGET PRODUCT NAME: {product_name}

Return `page_contains_target: true` if ANY of the following is true:
  - A row whose variant name / SKU code mentions "{product_name}" (case-insensitive, accent-insensitive, any language) is visible on this page
  - A shared technical-characteristics icon strip (MATT / GLOSS / SHADE VARIATION / SHOWER WALL / SHOWER FLOOR / FLOOR / TRAFFIC plus R-rating / PEI / water absorption / fire / frost) is visible and would apply to this product
  - A shared packing-table header is visible (UNIT / m² / PIECES / BOX / BOXES PALLET / WEIGHT PALLET etc.) even if the specific row data is mixed across products

ONLY return `page_contains_target: false` when the page is CLEARLY a different product's photo / intro / brand spread with nothing that could apply to "{product_name}".

For the fields you return:
  - variants / SKU codes / packing_per_variant / grout_recommendations → ONLY include rows whose variant name starts with or mentions "{product_name}" (case-insensitive). Drop rows for other products on the same table.
  - slip_resistance / pei_rating / water_absorption_class / fire_rating / frost_resistance / shade_variation / traffic_level / installation_method / joint_width_mm / certifications → include if visible on this page, even if presented as a shared spec grid. Those icons are usually one set per spec page.
  - thickness_mm / finish / body_type / dimensions → include if the page shows values specifically tied to "{product_name}".

TECHNICAL CHARACTERISTICS ICON STRIP — read carefully
-----------------------------------------------------
Ceramic catalogs show tech specs as a ROW of small square pictograms near the top or bottom of the spec page, each with a tiny label underneath. You MUST inspect every icon and extract its value when visible. The icons you will see in Harmony / similar catalogs:

  • MATT / GLOSS          — finish dot, report "matte" or "gloss" in `finish`
  • SHADE VARIATION       — V1 / V2 / V3 / V4 → `shade_variation`
  • SLIP RESISTANCE       — foot on wet surface, look for R9 / R10 / R11 / R12 label → `slip_resistance`
  • PEI                   — circle of arrows or roman numeral I..V → `pei_rating`
  • WATER ABSORPTION      — water droplet + BIa / BIb / BIIa / BIIb / BIII → `water_absorption_class`
  • FIRE RATING           — flame + A1 / A2 / Bfl / Cfl → `fire_rating`
  • FROST RESISTANCE      — snowflake; yes/no → `frost_resistance`
  • TRAFFIC LEVEL         — footprint; residential / commercial / heavy → `traffic_level`
  • SHOWER WALL / FLOOR / FLOOR — check marks next to each → include in `recommended_use` array

Most pages show 6-10 of these pictograms together. Even if a value looks subtle, RETURN WHAT YOU SEE rather than null. If a pictogram is clearly struck-through or dimmed, it's an absent feature — skip it. If you genuinely cannot read the icon strip, use null for that field but still populate the other fields on the page.

If page_contains_target is false, return exactly:
{"page_contains_target": false, "product_name": null}

Otherwise return STRICT JSON (no prose, no markdown fences). Fields you cannot clearly see should be null. Arrays you cannot populate should be []:
{
  "page_contains_target": true,
  "product_name": "{product_name}",
  "dimensions_cm": "...",
  "dimensions_inch": "...",
  "thickness_mm": null,
  "finish": null,
  "body_type": null,
  "patterns": [],
  "colors": [],
  "variants": [{"sku":"","name":"","color":"","format":"","pattern":""}],
  "pieces_per_box": null,
  "m2_per_box": null,
  "sqft_per_box": null,
  "weight_per_box_kg": null,
  "weight_per_box_lb": null,
  "boxes_per_pallet": null,
  "m2_per_pallet": null,
  "weight_per_pallet_kg": null,
  "weight_per_pallet_lb": null,
  "packaging_per_variant": [
    {
      "variant": "",
      "format": "",
      "pcs_box": null,
      "m2_box": null,
      "sqft_box": null,
      "weight_box_kg": null,
      "weight_box_lb": null,
      "boxes_pallet": null,
      "weight_pallet_kg": null
    }
  ],
  "slip_resistance": null,
  "pei_rating": null,
  "water_absorption_class": null,
  "water_absorption_pct": null,
  "fire_rating": null,
  "frost_resistance": null,
  "shade_variation": null,
  "traffic_level": null,
  "recommended_use": [],
  "grout_recommendations": [{"supplier":"","product":"","code":"","for_variant":""}],
  "certifications": [],
  "installation_method": null,
  "joint_width_mm": null
}

For packaging_per_variant: only include rows whose variant name starts with or matches "{product_name}". Drop every row that clearly belongs to a different product, even if it's on the same packing table.

The technical characteristics icons section is usually a strip of small pictograms labeled MATT, GLOSS, SHADE VARIATION, SHOWER WALL, SHOWER FLOOR, FLOOR, TRAFFIC, etc. Below or beside those labels look for values like R10, R11, PEI III, V2, BIa, A1, Class II — read each icon's state even if it's subtle.

CRITICAL rules:
- Use null for fields you cannot clearly see on this page.
- Do NOT hallucinate values. Empty > guessed.
- For arrays, return [] if empty.
- NEVER include variants, SKU codes, packing rows, or grout recommendations for a product OTHER than "{product_name}".
- Return JSON only. No prose. No markdown fences."""

def _build_spec_prompt(product_name: str) -> str:
    """Fill SPEC_PROMPT_TEMPLATE with the target product name.

    We use a plain `.replace` rather than `.format` because the template
    embeds literal JSON braces that would trip .format's {…} parser.
    """
    safe = (product_name or "").strip() or "the ceramic product on this page"
    return SPEC_PROMPT_TEMPLATE.replace("{product_name}", safe)


# Backwards-compat constant for any caller that still imports SPEC_PROMPT
# without a product name (emits a generic prompt).
SPEC_PROMPT = _build_spec_prompt("the ceramic product on this page")


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


def _call_claude_vision(png_bytes: bytes, prompt: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Single Claude Vision call, returns parsed JSON dict or None on failure."""
    if prompt is None:
        prompt = SPEC_PROMPT
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
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=CLAUDE_VISION_MODEL,
            max_tokens=3000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
    except Exception as e:
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
    """
    candidates = [
        f"/tmp/pdf_processor_{document_id}/{document_id}.pdf",
        f"/tmp/pdf_processor_{document_id}/source.pdf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _find_pdf_pages_by_text(
    pdf_path: str,
    product_name: str,
    max_pages: int = 12,
) -> List[int]:
    """Scan the PDF and return 0-indexed page indices whose text contains
    `product_name` (case- and accent-insensitive).

    This is the authoritative signal for "where does this product live in the
    PDF" when the chunk metadata's `product_pages` turns out to be catalog
    folio labels (two per physical spread) rather than absolute PDF indices.
    """
    import unicodedata

    def _normalize(s: str) -> str:
        # Strip accents + uppercase so "PIQUÉ" matches "PIQUE" and vice versa.
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        return s.upper().strip()

    needle = _normalize(product_name)
    if not needle:
        return []

    doc = fitz.open(pdf_path)
    matches: List[int] = []
    try:
        for i in range(doc.page_count):
            text = _normalize(doc[i].get_text())
            if needle in text:
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

    # Primary: name-based text scan
    text_matches: List[int] = []
    if product_name:
        text_matches = _find_pdf_pages_by_text(pdf_path, product_name)

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
        Merged spec dict (fields defined by SPEC_PROMPT_TEMPLATE schema).
        Empty on failure.
    """
    if not os.path.exists(pdf_path):
        logger.warning(f"product_spec_vision_extractor: PDF not found at {pdf_path}")
        return {}

    pdf_indices = _resolve_pdf_pages_for_product(
        pdf_path, product_page_range, product_name=product_name,
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
            data = _call_claude_vision(image_bytes, prompt=product_aware_prompt)
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
