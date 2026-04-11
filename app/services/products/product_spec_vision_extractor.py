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

# Render DPI for PDF pages. 220 keeps letterpress / icon strips legible while
# staying well under Claude's 5 MB input limit on large A4 pages. Anything
# higher and we frequently have to re-shrink — which wastes time and risks
# losing icon detail from the downscale pass.
PAGE_RENDER_DPI = 220

# Max bytes per image before we downscale. 4 MB leaves comfortable headroom
# under Claude's 5 MB hard limit after base64 expansion overhead.
MAX_IMAGE_BYTES = 4_000_000

# The single prompt we send for every product spec page. The schema is
# deliberately flat — nested objects make the post-parse merge harder.
# Placeholder {product_name} is substituted at call time so Claude can
# filter multi-product spec pages down to just the one we're enriching.
SPEC_PROMPT_TEMPLATE = """You are reading one page of a ceramic tile catalog. The page may contain spec data for MULTIPLE products. We are enriching ONLY this product:

    TARGET PRODUCT NAME: {product_name}

Your job is to extract technical specifications that apply ONLY to "{product_name}". Ignore data that clearly belongs to a different product on the same page (different product header / different variant-name prefix / different color family).

If the page contains NO data for "{product_name}" (e.g. it's a photo page for a different product, or a brand intro page), return this exact empty envelope:
{"page_contains_target": false, "product_name": null}

Otherwise return STRICT JSON with every field you CAN see for "{product_name}" and null for fields you cannot. Look carefully at:

1. Product name, variants, SKU codes, color labels — ONLY "{product_name}" variants
2. Dimensions (metric cm + imperial inch), thickness (mm)
3. Colors, finish, body type, patterns
4. Packing table: pieces/box, m²/box, sqft/box, boxes/pallet, weight/box (kg AND lb), m²/pallet, weight/pallet (kg AND lb)
5. Technical characteristics icons: slip resistance (R9/R10/R11/R12), PEI rating (PEI I/II/III/IV/V), water absorption class (BIa/BIb/BIIa/BIIb/BIII), fire rating (A1/A2/Bfl/Cfl), frost resistance (yes/no), shade variation (V1/V2/V3/V4), traffic level (residential/commercial/heavy)
6. Recommended use (shower wall, shower floor, wall, floor, outdoor)
7. Grout recommendations (per color: supplier + product name + color code)
8. Certifications (ISO, EN, CE, DIN, BS, ANSI, LEED)
9. Installation method, joint width

Return STRICT JSON ONLY (no prose, no markdown fences):
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


def _render_pdf_page_to_bytes(pdf_path: str, page_index: int, dpi: int = PAGE_RENDER_DPI) -> bytes:
    """Render one PDF page to PNG bytes."""
    doc = fitz.open(pdf_path)
    try:
        pix = doc[page_index].get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def _shrink_if_needed(png_bytes: bytes, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    """Downscale a PNG (and optionally flatten to JPEG) to fit under max_bytes.

    Guarantees a return value under max_bytes whenever possible. Tries PNG at
    progressively smaller sizes first; if PNG can't get small enough (very
    high-res brochure pages with many gradients compress poorly), falls back
    to JPEG at quality 85 which is ~4-6x more efficient for photographic
    catalog content. Icon glyphs and spec table text survive JPEG 85.
    """
    if len(png_bytes) <= max_bytes:
        return png_bytes
    im = Image.open(io.BytesIO(png_bytes))
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


def _resolve_pdf_pages_for_product(
    pdf_path: str,
    product_page_range: List[int],
) -> List[int]:
    """Map product page numbers to 0-indexed PDF page indices — deterministic.

    Input convention: `product_page_range` is a list of PDF page numbers as
    stored in `document_chunks.metadata.product_pages` — the MIVAA extraction
    pipeline writes absolute 1-indexed PDF page numbers there (NOT catalog
    page labels), so the conversion is simply `n - 1` to get 0-indexed
    PyMuPDF indices. No fuzzy offset guessing.

    The old implementation fanned each page out to four candidate offsets
    `(-2, -1, 0, 1)` and capped the result at the first 4 — which for a
    6-page range would shift the window *backwards* past where the spec
    table actually lives. That was the root cause of VALENOVA picking up
    the preceding product's spec data on the first backfill run.

    Args:
        pdf_path: Path to the PDF (used only to bounds-check page count).
        product_page_range: List of 1-indexed PDF page numbers.

    Returns:
        Sorted list of 0-indexed PyMuPDF page indices, clamped to document
        range. Empty list if input is empty or nothing is in range.
    """
    if not product_page_range:
        return []
    doc = fitz.open(pdf_path)
    total = doc.page_count
    doc.close()

    resolved = sorted({
        int(p) - 1
        for p in product_page_range
        if isinstance(p, (int, str)) and str(p).isdigit() and 0 <= int(p) - 1 < total
    })
    logger.info(
        f"   🗺  spec vision page map: input={sorted(product_page_range)} "
        f"→ 0-indexed PDF pages {resolved} (total={total})"
    )
    return resolved


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

    pdf_indices = _resolve_pdf_pages_for_product(pdf_path, product_page_range)
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
        try:
            png = _render_pdf_page_to_bytes(pdf_path, idx)
        except Exception as e:
            logger.warning(f"product_spec_vision_extractor: render page {idx} failed: {e}")
            continue

        data = _call_claude_vision(png, prompt=product_aware_prompt)
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
