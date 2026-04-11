"""
Product Spec Extractor v2 — Layer 3 of the reusable PDF spec pipeline.

Runs per product. For each product, extracts technical-characteristics and
packing specifications from the PDF using a 3-tier hybrid strategy:

  Tier A  PyMuPDF text-dict parser  (free, deterministic, exact values)
  Tier B  Claude Sonnet Vision      (fallback when Tier A is below threshold)
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

SONNET_FALLBACK_MODEL = os.getenv(
    "PRODUCT_SPEC_SONNET_MODEL",
    "claude-sonnet-4-6",
)

# Tier A coverage threshold — if PyMuPDF parser fills at least this many
# of the core packing fields, we skip the Sonnet fallback for cost savings.
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

# Column headers we recognize. Maps a normalized header label to the target
# field in the output schema. Order matters — we iterate the known row and
# attach each value to the nearest matching header by x-coordinate.
COLUMN_HEADERS: List[Tuple[str, str]] = [
    # Technical characteristics (icon-bullet columns)
    ("MATT",              "finish_matt"),
    ("GLOSS",             "finish_gloss"),
    ("SHADE VARIATION",   "shade_variation_flag"),
    ("SHOWER WALL",       "recommended_use_shower_wall"),
    ("SHOWER FLOOR",      "recommended_use_shower_floor"),
    ("FLOOR",             "recommended_use_floor"),
    ("TRAFFIC",           "traffic_flag"),
    # Packing numeric columns
    ("UNIT M2 PIECES",    "pieces_per_m2"),
    ("UNIT M2",           "pieces_per_m2"),
    ("PIECES M2",         "pieces_per_m2"),
    ("PIECES SQ F",       "pieces_per_sqft"),
    ("PIECES SQFT",       "pieces_per_sqft"),
    ("PIECES BOX",        "pieces_per_box"),
    ("PCS BOX",           "pieces_per_box"),
    ("BOX M2",            "m2_per_box"),
    ("M2 BOX",            "m2_per_box"),
    ("BOX SQ FT",         "sqft_per_box"),
    ("BOX SQFT",          "sqft_per_box"),
    ("SQ FT BOX",         "sqft_per_box"),
    ("WEIGHT BOX",        "weight_per_box_kg"),
    ("BOX WEIGHT",        "weight_per_box_kg"),
    ("WEIGHT BOX LB",     "weight_per_box_lb"),
    ("BOX WEIGHT LB",     "weight_per_box_lb"),
    ("BOXES PALLET",      "boxes_per_pallet"),
    ("PALLET BOXES",      "boxes_per_pallet"),
    ("M2 PALLET",         "m2_per_pallet"),
    ("PALLET M2",         "m2_per_pallet"),
    ("SQ FT PALLET",      "sqft_per_pallet"),
    ("PALLET SQ FT",      "sqft_per_pallet"),
    ("WEIGHT PALLET",     "weight_per_pallet_kg"),
    ("PALLET WEIGHT",     "weight_per_pallet_kg"),
    ("WEIGHT PALLET LB",  "weight_per_pallet_lb"),
    ("CM PALLET",         "pallet_dimensions_cm"),
    ("PALLET CM",         "pallet_dimensions_cm"),
    ("THICKNESS MM",      "thickness_mm"),
    ("THICKNESS INCH",    "thickness_inch"),
]


def _is_bullet_glyph(text: str) -> bool:
    """Common bullet characters used on ceramic spec tables to indicate
    'this column applies to this product'."""
    return text.strip() in {"•", "●", "◆", "◼", "■", "▪", "✓", "✔", "x", "X"}


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
    """Find the horizontal row of spans that starts with the product name.

    Strategy: locate the span whose text starts with the product name, then
    return every span within `y_tolerance` pixels of that span's center-y.
    Sorted left-to-right by x0.
    """
    n_name = _normalize(product_name)
    if not n_name:
        return []

    # Find the FIRST anchor span — typically the product-name label
    # printed on the same row as the numeric values. Prefer spans where
    # the span text starts with the product name (not just contains it).
    anchors = [s for s in spans if _normalize(s["text"]).startswith(n_name)]
    if not anchors:
        anchors = [s for s in spans if n_name in _normalize(s["text"])]
    if not anchors:
        return []

    # Pick the anchor with the smallest x (leftmost — usually the row label)
    anchor = min(anchors, key=lambda s: s["x0"])
    anchor_cy = anchor["cy"]

    row = [s for s in spans if abs(s["cy"] - anchor_cy) <= y_tolerance]
    row.sort(key=lambda s: s["x0"])
    return row


def _find_column_header_positions(
    spans: List[Dict[str, Any]],
) -> List[Tuple[str, float, float]]:
    """Find every column-header span and return (normalized_header, cx, cy).
    Header spans are the ones whose text matches one of the known header
    labels above. We search for multi-word headers by concatenating
    nearby spans on the same line."""
    lines: Dict[int, List[Dict[str, Any]]] = {}
    for s in spans:
        bucket = int(s["cy"] / 4)  # coarse Y bucketing for multi-line labels
        lines.setdefault(bucket, []).append(s)

    headers_found: List[Tuple[str, float, float]] = []
    for bucket, line_spans in lines.items():
        line_spans.sort(key=lambda s: s["x0"])
        # Concatenate adjacent spans to get multi-word header text
        i = 0
        while i < len(line_spans):
            for joined_count in (4, 3, 2, 1):
                if i + joined_count > len(line_spans):
                    continue
                joined = " ".join(line_spans[j]["text"] for j in range(i, i + joined_count))
                norm_joined = _normalize(joined)
                for header_text, _field in COLUMN_HEADERS:
                    if header_text == norm_joined:
                        # Use the cx of the first span in the joined set
                        cx = sum(line_spans[j]["cx"] for j in range(i, i + joined_count)) / joined_count
                        cy = sum(line_spans[j]["cy"] for j in range(i, i + joined_count)) / joined_count
                        headers_found.append((header_text, cx, cy))
                        i += joined_count
                        break
                else:
                    continue
                break
            else:
                i += 1
    return headers_found


def _map_row_to_headers(
    product_row: List[Dict[str, Any]],
    headers: List[Tuple[str, float, float]],
    x_tolerance: float = 40.0,
) -> Dict[str, Any]:
    """For each span in the product row, find the nearest column header by
    x-coordinate and assign the span's value to that column's target field.

    Returns a flat dict of {target_field: parsed_value}.
    """
    out: Dict[str, Any] = {}
    if not headers:
        return out

    # Build header cx → field lookup
    header_fields: Dict[str, str] = {h[0]: f for (h, f) in zip(headers, [COLUMN_HEADERS]) if False}
    # We need real lookup — iterate COLUMN_HEADERS once
    header_lookup = {h[0]: h[1] for h in COLUMN_HEADERS}

    for span in product_row:
        raw = span["text"].strip()
        if not raw:
            continue

        # Skip the anchor (product name) itself
        if _is_bullet_glyph(raw) or re.fullmatch(r"[-—–]+", raw):
            # Bullets: find which column they sit under and mark as flag
            best_header = None
            best_dist = float("inf")
            for header_text, cx, _cy in headers:
                d = abs(cx - span["cx"])
                if d < best_dist and d <= x_tolerance:
                    best_dist = d
                    best_header = header_text
            if best_header:
                field = header_lookup.get(best_header)
                if field and field.startswith(("finish_", "recommended_use_", "shade_variation_flag", "traffic_flag")):
                    out[field] = True
            continue

        # Numeric spans: parse and assign to nearest header
        value = _parse_number(raw)
        if value is None:
            # Non-numeric: might be dimensions like "120X80X91"
            if re.fullmatch(r"\d+\s*[xX]\s*\d+\s*[xX]\s*\d+", raw):
                value = raw.replace(" ", "")
            else:
                continue

        best_header = None
        best_dist = float("inf")
        for header_text, cx, _cy in headers:
            d = abs(cx - span["cx"])
            if d < best_dist and d <= x_tolerance:
                best_dist = d
                best_header = header_text
        if not best_header:
            continue
        field = header_lookup.get(best_header)
        if not field:
            continue
        # Don't overwrite an already-assigned field with a worse value
        if field not in out:
            out[field] = value

    return out


def _tier_a_pymupdf(
    pdf_path: str,
    page_indices: List[int],
    product_name: str,
) -> Dict[str, Any]:
    """Extract packing + flag fields from the product's spec pages via
    PyMuPDF text-dict parsing. Zero LLM cost, deterministic.

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
            headers = _find_column_header_positions(spans)
            if not headers:
                continue
            row_data = _map_row_to_headers(product_row, headers)
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
# Tier B — Claude Sonnet Vision fallback (reuses existing extractor)
# ──────────────────────────────────────────────────────────────────────────

def _tier_b_sonnet(
    pdf_path: str,
    page_indices: List[int],
    product_name: str,
) -> Dict[str, Any]:
    """Fallback: delegate to the existing product_spec_vision_extractor
    but with the Sonnet model override. Returns the same nested-metadata
    shape. Only runs when Tier A coverage is below threshold."""
    # Import locally to avoid a cyclic/test-time dependency
    from app.services.products import product_spec_vision_extractor as psve

    # Temporarily swap the model to Sonnet — thread-local is overkill here
    # since we call the function in-process and restore afterwards.
    original_model = psve.CLAUDE_VISION_MODEL
    psve.CLAUDE_VISION_MODEL = SONNET_FALLBACK_MODEL
    try:
        raw = psve.extract_specs_from_pdf_pages(
            pdf_path=pdf_path,
            product_page_range=[idx + 1 for idx in page_indices],  # 1-indexed input
            product_name=product_name,
        )
    except Exception as e:
        logger.warning(
            f"product_spec_extractor_v2: tier_b sonnet fallback failed: {e}"
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
        enable_tier_b: Skip Sonnet Vision fallback when False (useful for
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

        # ── Tier B — Sonnet Vision fallback ─────────────────────────────
        if enable_tier_b and tier_a_count < TIER_A_SUFFICIENT_FIELDS:
            logger.info(
                f"{log_prefix}: tier_a coverage below threshold "
                f"({tier_a_count} < {TIER_A_SUFFICIENT_FIELDS}), running tier_b sonnet"
            )
            tier_b_result = _tier_b_sonnet(pdf_path, page_indices, product_name)
            if tier_b_result:
                source_tiers.append("claude_sonnet_vision")
                merged = _merge_specs(merged, tier_b_result)
        elif tier_a_count >= TIER_A_SUFFICIENT_FIELDS:
            logger.info(
                f"{log_prefix}: tier_a sufficient ({tier_a_count} fields), skipping tier_b"
            )

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
