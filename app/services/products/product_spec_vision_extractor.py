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

# Render DPI for PDF pages. 300 is the sweet spot — any higher and the images
# exceed Claude's 5 MB input limit even after thumbnailing.
PAGE_RENDER_DPI = 300

# Max bytes per image before we downscale.
MAX_IMAGE_BYTES = 4_500_000

# The single prompt we send for every product spec page. The schema is
# deliberately flat — nested objects make the post-parse merge harder.
SPEC_PROMPT = """Extract ALL ceramic tile technical specifications visible on this page from the product specification PDF. Look carefully at:

1. Product name, variants, SKU codes, color labels
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
  "product_name": "...",
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

For packaging_per_variant: when the packing table shows ONE ROW PER FORMAT/VARIANT (different pieces/box, m²/box, or weight per format), capture every row. When a single packing spec covers the whole product, leave packaging_per_variant as [] and populate only the scalar pieces_per_box / m2_per_box / weight_* fields.

CRITICAL rules:
- Use null for fields you cannot clearly see on this page.
- Do NOT hallucinate values. Empty > guessed.
- For arrays, return [] if empty.
- For variants/grout, only include entries you can clearly read.
- Return JSON only. No prose. No markdown fences."""


def _render_pdf_page_to_bytes(pdf_path: str, page_index: int, dpi: int = PAGE_RENDER_DPI) -> bytes:
    """Render one PDF page to PNG bytes."""
    doc = fitz.open(pdf_path)
    try:
        pix = doc[page_index].get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def _shrink_if_needed(png_bytes: bytes, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    """Downscale a PNG to fit Claude's 5 MB limit while preserving spec legibility."""
    if len(png_bytes) <= max_bytes:
        return png_bytes
    im = Image.open(io.BytesIO(png_bytes))
    # Start at 2200 longest edge, halve if still too big
    for edge in (2200, 1800, 1400, 1000):
        scaled = im.copy()
        scaled.thumbnail((edge, edge))
        buf = io.BytesIO()
        scaled.save(buf, format="PNG", optimize=True)
        out = buf.getvalue()
        if len(out) <= max_bytes:
            return out
    return out  # last attempt even if still slightly over


def _call_claude_vision(png_bytes: bytes, prompt: str = SPEC_PROMPT) -> Optional[Dict[str, Any]]:
    """Single Claude Vision call, returns parsed JSON dict or None on failure."""
    if not ANTHROPIC_API_KEY:
        logger.error("product_spec_vision_extractor: ANTHROPIC_API_KEY not set")
        return None

    png_bytes = _shrink_if_needed(png_bytes)
    b64 = base64.b64encode(png_bytes).decode("utf-8")

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=CLAUDE_VISION_MODEL,
            max_tokens=3000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
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
    """Map catalog page numbers to 0-indexed PDF page indices.

    Catalog page numbering often has a 2-page offset from PDF page numbering
    (covers, TOC, etc. are unnumbered). We can't know the offset a priori
    without inspecting the PDF, so we try a few common mappings and pick the
    smallest set that's in range.

    Args:
        pdf_path: Path to the PDF.
        product_page_range: List of 1-indexed catalog page numbers.

    Returns:
        Best guess 0-indexed PDF page indices.
    """
    doc = fitz.open(pdf_path)
    total = doc.page_count
    doc.close()

    # Strategy: catalog page N usually maps to PDF index (N-1) or thereabouts.
    # For the Harmony signature book specifically, we saw VALENOVA on catalog
    # pages 24-31 = PDF indices 13-15 (roughly). So the actual relationship is
    # non-trivial. We send a subset anyway — worst case, Claude returns null
    # for all fields on a non-matching page and we move on.
    candidates = set()
    for p in product_page_range:
        # Try catalog = PDF, catalog = PDF+1, catalog = PDF-1 (common offsets)
        for offset in (-2, -1, 0, 1):
            idx = p - 1 + offset
            if 0 <= idx < total:
                candidates.add(idx)
    return sorted(candidates)


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
        product_page_range: 1-indexed catalog page numbers for this product.
        product_name: Optional — used for logging only.

    Returns:
        Merged spec dict (fields defined by SPEC_PROMPT schema). Empty on failure.
    """
    if not os.path.exists(pdf_path):
        logger.warning(f"product_spec_vision_extractor: PDF not found at {pdf_path}")
        return {}

    pdf_indices = _resolve_pdf_pages_for_product(pdf_path, product_page_range)
    if not pdf_indices:
        logger.info(f"product_spec_vision_extractor: no valid pages for {product_name}")
        return {}

    # Cap at 4 pages per product — prevents runaway cost on catalogs that assign
    # huge page ranges to a single product.
    pdf_indices = pdf_indices[:4]

    logger.info(
        f"📸 product_spec_vision_extractor: {product_name or '?'} "
        f"scanning {len(pdf_indices)} pages {pdf_indices}"
    )

    results: List[Dict[str, Any]] = []
    for idx in pdf_indices:
        try:
            png = _render_pdf_page_to_bytes(pdf_path, idx)
        except Exception as e:
            logger.warning(f"product_spec_vision_extractor: render page {idx} failed: {e}")
            continue

        data = _call_claude_vision(png)
        if data:
            results.append(data)
            # Quick break if we already got the main packing fields on an early page
            if all(
                data.get(k) not in (None, [], "")
                for k in ("pieces_per_box", "m2_per_box", "weight_per_box_kg", "boxes_per_pallet")
            ):
                logger.info(f"   ✅ full packing data found on PDF page {idx}, skipping remaining")
                break

    if not results:
        return {}

    merged = _select_best_spec_result(results)
    populated = sum(1 for v in merged.values() if v not in (None, [], ""))
    logger.info(
        f"   ✅ product_spec_vision_extractor: {populated}/{len(merged)} fields populated "
        f"from {len(results)} page(s)"
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
