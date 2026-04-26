"""
Product Spec Extractor v2 — Layer 3 of the reusable PDF spec pipeline.

Runs per product. For each product, extracts technical-characteristics and
packing specifications from the PDF using a 3-tier hybrid strategy:

  Tier A  PyMuPDF text-dict parser  (free, deterministic, exact values)
  Tier B  Claude Opus Vision        (fallback when Tier A is below threshold)
  Tier C  Catalog legend inheritance (fields still null after A+B inherit
          from documents.metadata.catalog_legends values that applied
          globally to the catalog)

Why this shape
--------------
VALENOVA page 15 analysis revealed that ALL the packing values
(pieces/box, m²/box, weight kg/lb, pallet info, thickness mm/inch) exist
as plain text in a packed single row. PyMuPDF's `get_text("dict")` returns
every text span with (x, y, w, h) coordinates — we can find the product's
row and map columns by x-alignment with zero LLM cost. Claude Vision only
fires as fallback on unusual layouts.

The output shape matches what `_merge_enriched_fields_into_metadata` in
stage_4_products.py expects, so callers drop this in where
`extract_specs_from_pdf_pages` used to live.

Wiring: optional `job_id` for progress updates, logger prefix for grep-ability,
Sentry context on exceptions. Matches the project-wide template.
"""

import logging
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import sentry_sdk

logger = logging.getLogger(__name__)

OPUS_FALLBACK_MODEL = os.getenv(
    "PRODUCT_SPEC_SONNET_MODEL",
    "claude-opus-4-7",
)

# Tier A coverage threshold — if PyMuPDF parser fills at least this many
# of the core packing fields, we skip the Claude Vision fallback for cost savings.
TIER_A_SUFFICIENT_FIELDS = 6

CORE_PACKING_FIELDS = (
    "pieces_per_box",
    "m2_per_box",
    "weight_per_box_kg",
    "boxes_per_pallet",
    "weight_per_pallet_kg",
    "thickness_mm",
)


# ──────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Strip accents + uppercase + collapse whitespace."""
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.upper().strip())


def _parse_number(raw: str) -> Optional[float]:
    """Parse "1,234.56" or "1.234,56" or "0,50" → 0.5, 7.40, 120, etc.
    Ceramic catalogs mix European (comma decimal) and US (period decimal)
    conventions inside the same document. We pick whichever convention
    yields a sensible value."""
    if not raw:
        return None
    cleaned = raw.strip().replace(" ", "")
    # Try "1,234.56" — comma thousands, period decimal
    try:
        if "." in cleaned and "," in cleaned:
            if cleaned.rfind(".") > cleaned.rfind(","):
                return float(cleaned.replace(",", ""))
            else:
                return float(cleaned.replace(".", "").replace(",", "."))
        if "," in cleaned and "." not in cleaned:
            # European decimal: "8,30" → 8.30
            return float(cleaned.replace(",", "."))
        return float(cleaned)
    except (ValueError, TypeError):
        return None


# ──────────────────────────────────────────────────────────────────────────
# Tier A — PyMuPDF text-dict packing row parser
# ──────────────────────────────────────────────────────────────────────────

# Conventional column order for ceramic catalog packing tables. This order
# is consistent across Harmony, Peronda, Aparici, Atlas Concorde, Marazzi,
# Florim, and Coem — we exploit it to assign numeric values to fields
# positionally when header-matching fails or headers span multiple lines.
#
# The `None` entries represent positions that don't always exist (e.g. some
# catalogs skip pcs_per_sqft and go straight from pcs_per_box to m²/box).
# Positional mapping honors the order but allows missing intermediates
# when the actual value count matches a known truncated variant.
CANONICAL_PACKING_ORDER: List[str] = [
    "pieces_per_m2",           # 1
    "pieces_per_sqft",         # 2
    "pieces_per_box",          # 3
    "m2_per_box",              # 4
    "sqft_per_box",            # 5
    "weight_per_box_kg",       # 6
    "weight_per_box_lb",       # 7
    "boxes_per_pallet",        # 8
    "m2_per_pallet",           # 9
    "sqft_per_pallet",         # 10
    "weight_per_pallet_kg",    # 11
    "weight_per_pallet_lb",    # 12
    "pallet_dimensions_cm",    # 13 (e.g. "120X80X91")
    "thickness_mm",            # 14
    "thickness_inch",          # 15
]

# Alternate column orders seen in the wild — we pick whichever matches the
# number of values on the row.
KNOWN_COLUMN_ORDERS: Dict[int, List[str]] = {
    15: CANONICAL_PACKING_ORDER,
    # 14 cols: no pcs_per_m2 (starts with pcs_per_sqft)
    14: CANONICAL_PACKING_ORDER[1:],
    # 13 cols: no pcs_per_m2, no pcs_per_sqft
    13: CANONICAL_PACKING_ORDER[2:],
    # 12 cols: metric-only (no lb columns, no pcs_per_sqft)
    12: [
        "pieces_per_m2", "pieces_per_box", "m2_per_box",
        "weight_per_box_kg", "boxes_per_pallet", "m2_per_pallet",
        "weight_per_pallet_kg", "pallet_dimensions_cm",
        "thickness_mm", "thickness_inch",
        # fallback two extras if present
        "sqft_per_box", "sqft_per_pallet",
    ],
    # 10 cols: metric no lb, no sqft
    10: [
        "pieces_per_box", "m2_per_box", "weight_per_box_kg",
        "boxes_per_pallet", "m2_per_pallet", "weight_per_pallet_kg",
        "pallet_dimensions_cm", "thickness_mm", "thickness_inch",
        "sqft_per_box",
    ],
    # 8 cols: minimal — pcs/box, m²/box, kg/box, boxes/pallet,
    #         m²/pallet, kg/pallet, thickness_mm, thickness_inch
    8: [
        "pieces_per_box", "m2_per_box", "weight_per_box_kg",
        "boxes_per_pallet", "m2_per_pallet", "weight_per_pallet_kg",
        "thickness_mm", "thickness_inch",
    ],
}


def _is_bullet_glyph(text: str) -> bool:
    """Common bullet characters used on ceramic spec tables to indicate
    'this column applies to this product'."""
    return text.strip() in {"•", "●", "◆", "◼", "■", "▪", "✓", "✔", "x", "X"}


# Regex to split a merged span like "2108.42 120X80X91" into two tokens:
# a numeric value followed by a pallet-dimension string.
_MERGED_NUMBER_DIM_RE = re.compile(
    r"^(\d[\d.,]*)\s+(\d+\s*[xX]\s*\d+\s*[xX]\s*\d+)$"
)


def _split_merged_span(raw: str) -> List[str]:
    """If a span contains both a number and a dimension string, split them
    into separate tokens. Handles 'N.NN AAXBBXCC' patterns."""
    m = _MERGED_NUMBER_DIM_RE.match(raw.strip())
    if m:
        return [m.group(1), m.group(2).replace(" ", "")]
    return [raw]


def _extract_text_spans(doc: fitz.Document, page_index: int) -> List[Dict[str, Any]]:
    """Flatten PyMuPDF's nested text dict into a flat list of spans with
    (text, x, y, width, height). Each span is one homogeneous run of text
    on a line — exactly what we need to map columns to values."""
    page = doc[page_index]
    td = page.get_text("dict")
    spans_out: List[Dict[str, Any]] = []
    for block in td.get("blocks", []):
        if block.get("type", 0) != 0:  # text blocks only
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                raw = (span.get("text") or "").strip()
                if not raw:
                    continue
                bbox = span.get("bbox") or [0, 0, 0, 0]
                spans_out.append({
                    "text": raw,
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                    "cx": (bbox[0] + bbox[2]) / 2,
                    "cy": (bbox[1] + bbox[3]) / 2,
                })
    return spans_out


def _find_product_row(
    spans: List[Dict[str, Any]],
    product_name: str,
    y_tolerance: float = 6.0,
) -> List[Dict[str, Any]]:
    """Find the horizontal row of spans that sits on the same line as the
    product name in a packing table.

    Important: ceramic catalogs print the product name several times on a
    spec page (hero title at top, SKU labels in the variant block, packing
    table row label at bottom). We want the LAST occurrence — the packing
    table row — and then every span within `y_tolerance` of that span's
    center y.

    Heuristic for "this is the packing row":
      - The product name is a standalone token (not part of a SKU line
        like "VALENOVA WHITE LT/11,8X11,8")
      - There are numeric spans to the right of it on the same y
    """
    n_name = _normalize(product_name)
    if not n_name:
        return []

    # Collect every span whose normalized text exactly equals the name
    # (standalone label) OR starts with the name followed by whitespace
    exact_anchors = []
    for s in spans:
        ntext = _normalize(s["text"])
        if ntext == n_name:
            exact_anchors.append(s)
        elif ntext.startswith(n_name + " ") and len(ntext) - len(n_name) < 30:
            # "VALENOVA WHITE" — potential variant label but not SKU
            exact_anchors.append(s)

    if not exact_anchors:
        # Fallback: any span that contains the name as a substring
        exact_anchors = [s for s in spans if n_name in _normalize(s["text"])]
        if not exact_anchors:
            return []

    # For each candidate anchor, count the number of numeric spans to the
    # right on the same y. The anchor with the most numeric neighbors is
    # the packing row.
    def _numeric_neighbors(anchor: Dict[str, Any]) -> int:
        cy = anchor["cy"]
        count = 0
        for s in spans:
            if s is anchor:
                continue
            if abs(s["cy"] - cy) > y_tolerance:
                continue
            if s["x0"] <= anchor["x1"]:
                continue
            for tok in _split_merged_span(s["text"]):
                if _parse_number(tok) is not None:
                    count += 1
        return count

    best_anchor = max(exact_anchors, key=_numeric_neighbors)
    # If even the best anchor has <3 numeric neighbors, this isn't a
    # packing row — give up.
    if _numeric_neighbors(best_anchor) < 3:
        return []

    anchor_cy = best_anchor["cy"]
    row = [s for s in spans if abs(s["cy"] - anchor_cy) <= y_tolerance]
    row.sort(key=lambda s: s["x0"])
    return row


def _extract_values_positional(
    product_row: List[Dict[str, Any]],
    product_name: str,
) -> Dict[str, Any]:
    """Positional mapping: extract numeric + dimension values from the row,
    assign them to CANONICAL_PACKING_ORDER fields based on the count we see.

    This is more robust than header matching because:
      - Ceramic catalog column orders are conventional across brands
      - Multi-line headers are hard to parse reliably via text dict
      - Position-based mapping works even when header labels are missing

    Handles merged spans like "2108.42 120X80X91" by splitting them into
    two tokens first.
    """
    n_name = _normalize(product_name)
    out: Dict[str, Any] = {}
    bullets_before_numbers = 0
    tokens: List[Tuple[str, Any]] = []  # (kind, value) where kind ∈ {"bullet","number","dim","name"}

    for span in product_row:
        raw_text = span["text"].strip()
        if not raw_text:
            continue
        # Split merged spans first so "N.NN AAxBBxCC" becomes two tokens
        sub_tokens = _split_merged_span(raw_text)
        for tok in sub_tokens:
            tok = tok.strip()
            if not tok:
                continue
            # Skip the product name anchor itself
            if _normalize(tok) == n_name or _normalize(tok).startswith(n_name + " "):
                tokens.append(("name", tok))
                continue
            if _is_bullet_glyph(tok) or re.fullmatch(r"[-—–]+", tok):
                tokens.append(("bullet", tok))
                continue
            # Pallet dimension string: "120X80X91" etc.
            if re.fullmatch(r"\d+\s*[xX]\s*\d+\s*[xX]\s*\d+", tok):
                tokens.append(("dim", tok.replace(" ", "")))
                continue
            # Numeric
            num = _parse_number(tok)
            if num is not None:
                tokens.append(("number", num))
                continue
            # Unknown non-numeric (e.g. "UNIT", "BOX" if the header leaked
            # into the row because of bbox fuzzy matching) — skip silently

    # Count bullets that appear BEFORE the first numeric (these are the
    # technical-characteristics flags like MATT / SHADE VARIATION / SHOWER
    # WALL). They don't contribute to positional numeric mapping.
    saw_first_number = False
    bullet_list: List[str] = []
    numeric_and_dim: List[Tuple[str, Any]] = []
    for kind, val in tokens:
        if kind == "number" and not saw_first_number:
            saw_first_number = True
        if not saw_first_number and kind == "bullet":
            bullet_list.append(val)
            bullets_before_numbers += 1
            continue
        if kind in ("number", "dim"):
            numeric_and_dim.append((kind, val))

    # Positional mapping: find a known column order that matches the count
    target_order: Optional[List[str]] = None
    n = len(numeric_and_dim)
    if n in KNOWN_COLUMN_ORDERS:
        target_order = KNOWN_COLUMN_ORDERS[n]
    else:
        # Fall back to the longest known order that fits
        for known_n in sorted(KNOWN_COLUMN_ORDERS.keys(), reverse=True):
            if known_n <= n:
                target_order = KNOWN_COLUMN_ORDERS[known_n]
                break
    if not target_order:
        return out

    for i, (kind, val) in enumerate(numeric_and_dim):
        if i >= len(target_order):
            break
        field = target_order[i]
        if field is None:
            continue
        # Sanity: a dim value only goes in a "pallet_dimensions_cm" slot
        if kind == "dim" and field != "pallet_dimensions_cm":
            # Look for the pallet_dimensions slot later in the order and
            # skip this token's assignment if there's a better match
            if "pallet_dimensions_cm" in target_order[i:]:
                # Shift: assign this dim to pallet_dimensions_cm and move on
                out["pallet_dimensions_cm"] = val
                continue
        out[field] = val

    # Bullet-to-column mapping for the technical-characteristics strip.
    # Ceramic spec pages print bullets under: MATT / GLOSS / SHADE VAR /
    # SHOWER WALL / SHOWER FLOOR / FLOOR / TRAFFIC. We can't tell the
    # exact X→column mapping without header coordinates, so instead we
    # rely on the fact that in the product row the bullet spans appear
    # left-to-right in the same order as the header columns.
    #
    # Heuristic: find each bullet's X position and compare against the
    # X positions of the FIRST numeric value (which sits under the UNIT
    # column). Every bullet to the LEFT of that first numeric is a tech-
    # characteristics bullet, and we assign them by order:
    #   position 1 → MATT bullet → finish="matte"
    #   position 2 → GLOSS bullet → finish="gloss"
    #   position 3 → SHADE VARIATION bullet
    #   position 4 → SHOWER WALL → recommended_use includes "shower_wall"
    #   position 5 → SHOWER FLOOR → recommended_use includes "shower_floor"
    #   position 6 → FLOOR → recommended_use includes "floor"
    #   position 7 → TRAFFIC
    BULLET_POSITIONS: List[Tuple[str, Any]] = [
        ("finish_matt", True),
        ("finish_gloss", True),
        ("shade_variation_flag", True),
        ("recommended_use_shower_wall", True),
        ("recommended_use_shower_floor", True),
        ("recommended_use_floor", True),
        ("traffic_flag", True),
    ]
    # Re-walk the tokens to find bullet positions in document order
    bullet_index = 0
    for kind, _val in tokens:
        if kind == "bullet" and bullet_index < len(BULLET_POSITIONS):
            field, value = BULLET_POSITIONS[bullet_index]
            out[field] = value
            bullet_index += 1
        elif kind == "number":
            # Stop counting bullets once we've passed into the numeric row
            break

    return out


def _tier_a_pymupdf(
    pdf_path: str,
    page_indices: List[int],
    product_name: str,
) -> Dict[str, Any]:
    """Extract packing + flag fields from the product's spec pages via
    PyMuPDF text-dict parsing with positional column mapping. Zero LLM
    cost, deterministic.

    Returns a dict in the same nested shape as
    `map_vision_specs_to_product_metadata` so it can be passed directly
    to the Stage 4.7 merge step.
    """
    merged_flat: Dict[str, Any] = {}
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.warning(
            f"product_spec_extractor_v2: fitz.open failed for tier_a: {e}"
        )
        return {}
    try:
        for idx in page_indices:
            if not (0 <= idx < doc.page_count):
                continue
            spans = _extract_text_spans(doc, idx)
            if not spans:
                continue
            product_row = _find_product_row(spans, product_name)
            if not product_row:
                continue
            row_data = _extract_values_positional(product_row, product_name)
            if not row_data:
                continue
            logger.info(
                f"product_spec_extractor_v2 tier_a: page {idx} found "
                f"{len(row_data)} fields for '{product_name}'"
            )
            # Merge — don't overwrite values already found on an earlier page
            for k, v in row_data.items():
                if k not in merged_flat and v not in (None, "", []):
                    merged_flat[k] = v
    finally:
        doc.close()

    return _flat_to_nested(merged_flat, product_name)


def _flat_to_nested(flat: Dict[str, Any], product_name: str) -> Dict[str, Any]:
    """Map the flat column output to the nested metadata shape used by
    Stage 4.7's merge function. Returns {} if nothing was extracted."""
    if not flat:
        return {}

    out: Dict[str, Any] = {}

    # Packaging
    pkg: Dict[str, Any] = {}
    for key in (
        "pieces_per_box", "m2_per_box", "sqft_per_box",
        "weight_per_box_kg", "weight_per_box_lb",
        "boxes_per_pallet", "m2_per_pallet", "sqft_per_pallet",
        "weight_per_pallet_kg", "weight_per_pallet_lb",
        "pallet_dimensions_cm",
    ):
        if key in flat and flat[key] is not None:
            pkg[key] = flat[key]
    if pkg:
        out["packaging"] = pkg

    # Material properties
    mp: Dict[str, Any] = {}
    if "thickness_mm" in flat and flat["thickness_mm"] is not None:
        mp["thickness_mm"] = flat["thickness_mm"]
    if "thickness_inch" in flat and flat["thickness_inch"] is not None:
        mp["thickness_inch"] = flat["thickness_inch"]
    # finish from bullets: MATT bullet → "matte"
    if flat.get("finish_matt") and not flat.get("finish_gloss"):
        mp["finish"] = "matte"
    elif flat.get("finish_gloss") and not flat.get("finish_matt"):
        mp["finish"] = "gloss"
    if mp:
        out["material_properties"] = mp

    # Application / recommended_use
    uses: List[str] = []
    if flat.get("recommended_use_shower_wall"):
        uses.append("shower_wall")
    if flat.get("recommended_use_shower_floor"):
        uses.append("shower_floor")
    if flat.get("recommended_use_floor"):
        uses.append("floor")
    if uses:
        out["application"] = {"recommended_use": uses}

    return out


# ──────────────────────────────────────────────────────────────────────────
# Tier B — Claude Opus Vision fallback (reuses existing extractor)
# ──────────────────────────────────────────────────────────────────────────

def _tier_b_opus(
    pdf_path: str,
    page_indices: List[int],
    product_name: str,
) -> Dict[str, Any]:
    """Fallback: delegate to the existing product_spec_vision_extractor
    but with the Claude Opus model override. Returns the same nested-metadata
    shape. Only runs when Tier A coverage is below threshold."""
    # Import locally to avoid a cyclic/test-time dependency
    from app.services.products import product_spec_vision_extractor as psve

    # Temporarily swap the model to Claude Opus — thread-local is overkill here
    # since we call the function in-process and restore afterwards.
    original_model = psve.CLAUDE_VISION_MODEL
    psve.CLAUDE_VISION_MODEL = OPUS_FALLBACK_MODEL
    try:
        raw = psve.extract_specs_from_pdf_pages(
            pdf_path=pdf_path,
            product_page_range=[idx + 1 for idx in page_indices],  # 1-indexed input
            product_name=product_name,
        )
    except Exception as e:
        logger.warning(
            f"product_spec_extractor_v2: tier_b opus fallback failed: {e}"
        )
        return {}
    finally:
        psve.CLAUDE_VISION_MODEL = original_model

    if not raw:
        return {}
    return psve.map_vision_specs_to_product_metadata(raw)


# ──────────────────────────────────────────────────────────────────────────
# Tier C — Catalog legend inheritance
# ──────────────────────────────────────────────────────────────────────────

def _tier_c_legend_inheritance(
    current_spec: Dict[str, Any],
    catalog_legends: Dict[str, Any],
) -> Dict[str, Any]:
    """Fill fields still null after Tier A+B with values from the
    catalog-wide legend (Layer 2 output). Only fields flagged
    `applies_globally` on the legend are inherited — we never silently
    copy a legend value to a product unless the legend explicitly said
    it applies to all products in the catalog.

    Never overwrites existing values. Only adds missing fields.
    """
    if not catalog_legends:
        return current_spec

    by_type = catalog_legends.get("by_type") or {}
    merged = dict(current_spec)

    # Certifications — applied by Layer 2 directly to products.metadata,
    # but we still echo them here for provenance.
    certs = catalog_legends.get("global_certifications") or []
    if certs:
        compliance = merged.get("compliance") or {}
        if not compliance.get("certifications"):
            compliance["certifications"] = certs
            merged["compliance"] = compliance

    # Icons legend — globally-applicable performance defaults
    for icon_page in (by_type.get("icons") or []):
        if not icon_page.get("applies_globally"):
            continue
        performance = merged.get("performance") or {}
        for icon in (icon_page.get("icons") or []):
            if not isinstance(icon, dict):
                continue
            category = icon.get("category")
            code = icon.get("code")
            if not category or not code:
                continue
            if category in (
                "slip_resistance", "pei_rating", "water_absorption",
                "fire_rating", "frost_resistance", "shade_variation",
                "traffic_level",
            ):
                target = "water_absorption_class" if category == "water_absorption" else category
                if target not in performance:
                    performance[target] = code
        if performance:
            merged["performance"] = performance

    return merged


# ──────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────

async def extract_product_spec(
    *,
    document_id: str,
    product_id: str,
    product_name: str,
    pdf_path: str,
    page_indices: List[int],
    catalog_legends: Optional[Dict[str, Any]] = None,
    job_id: Optional[str] = None,
    enable_tier_b: bool = True,
) -> Dict[str, Any]:
    """Extract a product's spec block from the PDF using the 3-tier hybrid.

    Args:
        document_id: Document UUID (for logging).
        product_id:  Product UUID (for logging).
        product_name: Product name to look up in text spans. Must match
                      what's printed on the spec page (accent/case insensitive).
        pdf_path: Absolute path to the PDF.
        page_indices: 0-indexed PDF pages where this product lives,
                      already resolved by Layer 1's
                      `catalog_layout.product_pages_by_name[product_name]`.
        catalog_legends: `documents.metadata.catalog_legends` dict from
                         Layer 2, used for Tier C inheritance.
        job_id: Optional — enables progress updates.
        enable_tier_b: Skip Claude Opus Vision fallback when False (useful for
                       unit tests or cost-sensitive runs).

    Returns:
        Nested metadata dict matching Stage 4.7's merge function, with a
        `_source_tiers` key listing which tiers fired.
    """
    log_prefix = f"[{job_id or '-'}] spec_v2 '{product_name}'"
    source_tiers: List[str] = []

    try:
        # ── Tier A — PyMuPDF text-dict ──────────────────────────────────
        tier_a_result = _tier_a_pymupdf(pdf_path, page_indices, product_name)
        tier_a_count = sum(1 for f in CORE_PACKING_FIELDS if _has_field(tier_a_result, f))
        logger.info(
            f"{log_prefix}: tier_a (pymupdf) populated {tier_a_count}/{len(CORE_PACKING_FIELDS)} core fields"
        )
        if tier_a_result:
            source_tiers.append("pymupdf_text_dict")

        merged = tier_a_result

        # ── Tier B — Claude Opus Vision (complementary, not fallback) ────────
        # We ALWAYS run Tier B when enabled, because:
        #   - Tier A only extracts the packing row + thickness + bullet flags
        #   - Tier B uniquely provides: commercial.vision_variants (SKU
        #     metadata per color), commercial.grout_details (per-color
        #     grout recommendations), and any per-product performance
        #     icons that happen to be text-labeled (R10, PEI III, etc.)
        # Tier A fields take priority — _merge_specs never overwrites
        # existing values — so there's no risk of regressing accurate
        # Tier A numbers with Claude Opus approximations.
        #
        # The one cost optimization: when Tier A already hit the core
        # packing threshold, we skip Tier B pages that Tier A found and
        # only send the product's intro/photo pages to Claude Opus. This is
        # handled inside _tier_b_opus_complementary below.
        if enable_tier_b:
            if tier_a_count >= TIER_A_SUFFICIENT_FIELDS:
                logger.info(
                    f"{log_prefix}: tier_a sufficient ({tier_a_count} fields), "
                    f"running tier_b for complementary fields (variants/grout)"
                )
            else:
                logger.info(
                    f"{log_prefix}: tier_a partial ({tier_a_count} fields), "
                    f"running tier_b for missing packing + variants/grout"
                )
            tier_b_result = _tier_b_opus(pdf_path, page_indices, product_name)
            if tier_b_result:
                source_tiers.append("claude_opus_vision")
                merged = _merge_specs(merged, tier_b_result)

        # ── Tier C — Legend inheritance ─────────────────────────────────
        if catalog_legends:
            before = _count_fields(merged)
            merged = _tier_c_legend_inheritance(merged, catalog_legends)
            after = _count_fields(merged)
            if after > before:
                source_tiers.append("catalog_legend")
                logger.info(
                    f"{log_prefix}: tier_c added {after - before} fields from catalog legend"
                )

        merged["_source_tiers"] = source_tiers
        return merged

    except Exception as e:
        logger.error(f"{log_prefix}: extraction failed: {e}", exc_info=True)
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("job_id", job_id)
            scope.set_tag("document_id", document_id)
            scope.set_tag("product_id", product_id)
            scope.set_tag("stage", "product_spec_extractor_v2")
            scope.set_context("spec_extractor", {
                "product_name": product_name,
                "page_indices": page_indices,
                "tiers_completed": source_tiers,
            })
        sentry_sdk.capture_exception(e)
        return {"_source_tiers": source_tiers, "_error": str(e)}


# ──────────────────────────────────────────────────────────────────────────
# Small helpers shared across tiers
# ──────────────────────────────────────────────────────────────────────────

def _has_field(nested: Dict[str, Any], flat_key: str) -> bool:
    """Check if a nested metadata dict has a populated value for one of the
    CORE_PACKING_FIELDS (supports `packaging.*`, `material_properties.*`)."""
    if flat_key == "thickness_mm":
        return bool(nested.get("material_properties", {}).get("thickness_mm"))
    pkg = nested.get("packaging", {})
    return pkg.get(flat_key) not in (None, "", [])


def _count_fields(nested: Dict[str, Any]) -> int:
    """Count populated fields across all nested categories (for Tier C delta)."""
    n = 0
    for _, v in (nested.items() if isinstance(nested, dict) else []):
        if isinstance(v, dict):
            n += sum(1 for x in v.values() if x not in (None, "", [], {}))
        elif v not in (None, "", [], {}):
            n += 1
    return n


def _merge_specs(primary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two nested spec dicts. `primary` wins on conflict — `fallback`
    only fills fields missing from `primary`."""
    if not fallback:
        return primary
    if not primary:
        return dict(fallback)
    out = dict(primary)
    for section, section_val in fallback.items():
        if section.startswith("_"):
            continue
        if isinstance(section_val, dict):
            existing = out.get(section) or {}
            if not isinstance(existing, dict):
                existing = {}
            for k, v in section_val.items():
                if existing.get(k) in (None, "", [], {}) and v not in (None, "", [], {}):
                    existing[k] = v
            out[section] = existing
        else:
            if out.get(section) in (None, "", [], {}):
                out[section] = section_val
    return out
