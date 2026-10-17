"""Product Spec Extractor v2 — Layer 3 of the reusable PDF spec pipeline.

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
from app.services.metadata.field_registry import field_registry
from app.services.utilities.prompt_registry import prefetch

logger = logging.getLogger(__name__)

OPUS_FALLBACK_MODEL = os.getenv(
    "PRODUCT_SPEC_SONNET_MODEL",
    "claude-opus-5",
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
CANONICAL_PACKING_ORDER: List[str] = [
    "pieces_per_m2",           # 1
    "pieces_per_sqft",         # 2
    "pieces_per_box",          # 3
    "m2_per_box",              # 4
    "box_area_sqft",           # 5
    "weight_per_box_kg",       # 6
    "box_weight_lb",           # 7
    "boxes_per_pallet",        # 8
    "m2_per_pallet",           # 9
    "pallet_area_sqft",        # 10
    "weight_per_pallet_kg",    # 11
    "pallet_weight_lb",        # 12
    "pallet_dimensions_cm",    # 13 (e.g. "120X80X91")
    "thickness_mm",            # 14
    "thickness_in",            # 15
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
        "thickness_mm", "thickness_in",
        # fallback two extras if present
        "box_area_sqft", "pallet_area_sqft",
    ],
    # 10 cols: metric no lb, no sqft
    10: [
        "pieces_per_box", "m2_per_box", "weight_per_box_kg",
        "boxes_per_pallet", "m2_per_pallet", "weight_per_pallet_kg",
        "pallet_dimensions_cm", "thickness_mm", "thickness_in",
        "box_area_sqft",
    ],
    # 8 cols: minimal — pcs/box, m²/box, kg/box, boxes/pallet,
    #         m²/pallet, kg/pallet, thickness_mm, thickness_in
    8: [
        "pieces_per_box", "m2_per_box", "weight_per_box_kg",
        "boxes_per_pallet", "m2_per_pallet", "weight_per_pallet_kg",
        "thickness_mm", "thickness_in",
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


#: Header label -> flag field, matched against a header COLUMN's stacked text. Order
#: matters: the shower columns must claim their text before the plain floor column does.
_FLAG_HEADERS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("recommended_use_shower_wall", re.compile(r"SHOWER.*WALL|WALL.*SHOWER")),
    ("recommended_use_shower_floor", re.compile(r"SHOWER.*FLOOR|FLOOR.*SHOWER")),
    ("finish_matt", re.compile(r"\bMATT?E?\b")),
    ("finish_gloss", re.compile(r"GLOSS")),
    ("shade_variation_flag", re.compile(r"SHADE")),
    ("traffic_flag", re.compile(r"TRAFFIC")),
    ("recommended_use_floor", re.compile(r"\bFLOOR\b")),
]


def _flag_columns(
    page_spans: List[Dict[str, Any]],
    row_cy: float,
    max_height: float = 300.0,
    x_gap: float = 12.0,
) -> List[Tuple[float, float, str]]:
    """Header columns above the row that name a flag: (x0, x1, field).

    Header labels are often stacked over two lines ("SHOWER" / "WALL"), so spans above
    the row are grouped into columns by horizontal overlap before they are read.
    """
    above = [
        sp for sp in page_spans
        if row_cy - max_height < sp["cy"] < row_cy - 3
        and _parse_number(sp["text"]) is None
        and not _is_bullet_glyph(sp["text"].strip())
    ]
    columns: List[Dict[str, Any]] = []
    for sp in sorted(above, key=lambda x: (x["cx"], x["cy"])):
        for col in columns:
            if sp["x0"] <= col["x1"] + x_gap and sp["x1"] >= col["x0"] - x_gap \
                    and abs(sp["cx"] - col["cx"]) <= max(col["x1"] - col["x0"], 20):
                col["spans"].append(sp)
                col["x0"], col["x1"] = min(col["x0"], sp["x0"]), max(col["x1"], sp["x1"])
                break
        else:
            columns.append({"x0": sp["x0"], "x1": sp["x1"], "cx": sp["cx"], "spans": [sp]})

    out: List[Tuple[float, float, str]] = []
    claimed: set = set()
    for col in columns:
        text = " ".join(s["text"] for s in sorted(col["spans"], key=lambda x: x["cy"])).upper()
        for field, pattern in _FLAG_HEADERS:
            if field not in claimed and pattern.search(text):
                out.append((col["x0"], col["x1"], field))
                claimed.add(field)
                break
    return out


def _extract_values_positional(
    product_row: List[Dict[str, Any]],
    product_name: str,
    page_spans: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Positional mapping: extract numeric + dimension values from the row,
    assign them to CANONICAL_PACKING_ORDER fields based on the count we see.
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
    for i, (kind, val) in enumerate(numeric_and_dim if target_order else []):
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

    # A bullet sets the flag of the header COLUMN it sits under. Bullet order says nothing:
    # a row that marks only GLOSS has one bullet, and counting would call it MATT. No header
    # geometry means no flags, recorded as unmapped rather than guessed.
    bullet_spans = [sp for sp in product_row if _is_bullet_glyph(sp["text"].strip())]
    if bullet_spans:
        columns = _flag_columns(page_spans or [], product_row[0]["cy"]) if page_spans else []
        unmapped = 0
        for sp in bullet_spans:
            hits = [field for x0, x1, field in columns if x0 - 6 <= sp["cx"] <= x1 + 6]
            if len(hits) == 1:
                out[hits[0]] = True
            else:
                unmapped += 1
        if unmapped:
            out["_flags_unmapped"] = unmapped
            logger.info(f"product_spec_extractor_v2: {unmapped} flag bullet(s) for "
                        f"'{product_name}' sit under no recognised header column; left unset")

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
    image_only_pages: List[int] = []
    pages_scanned = 0
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        # M3-15 (#16): a bare {} here is indistinguishable from "this page had
        # no specs on it". The marker keeps the two apart for the caller and in
        # the _source_tiers audit trail (pipeline convention 1).
        logger.warning(
            f"product_spec_extractor_v2: fitz.open failed for tier_a: {e}"
        )
        return {"_tier_error": f"tier_a_open_failed: {str(e)[:200]}"}
    try:
        for idx in page_indices:
            if not (0 <= idx < doc.page_count):
                continue
            pages_scanned += 1
            spans = _extract_text_spans(doc, idx)
            if not spans:
                # No positional text spans at all → image-only / scanned page.
                # Tier A is STRUCTURALLY unable to read it: its packing-column
                # mapping needs per-token bbox geometry that a rendered image
                # doesn't expose, and the PaddleOCR cache carries reading-order
                # text but not per-token spans, so there's no cache fallback here
                # (unlike the plain-text consumers).
                image_only_pages.append(idx)
                continue
            product_row = _find_product_row(spans, product_name)
            if not product_row:
                continue
            row_data = _extract_values_positional(product_row, product_name, spans)
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

    # Make Tier A's structural blindness on image-only pages explicit (rather
    # than a silent zero), so operators can see when Tier B vision is carrying
    # the whole product instead of mistaking it for "no spec data exists".
    if image_only_pages and not merged_flat:
        logger.info(
            f"product_spec_extractor_v2 tier_a: no text spans on "
            f"{len(image_only_pages)}/{pages_scanned} page(s) for '{product_name}' "
            f"(image-only/scanned: {image_only_pages}) — deferring entirely to Tier B vision"
        )

    return _flat_to_nested(merged_flat, product_name)


#: Legend icon category -> performance key; a retired spelling resolves through the registry.
_ICON_CATEGORY_TARGETS = {
    "slip_resistance": "slip_resistance", "pei_rating": "pei_rating",
    "water_absorption_class": "water_absorption_class", "fire_rating": "fire_rating",
    "frost_resistance": "frost_resistance", "v_rating": "v_rating",
    "traffic_rating": "traffic_rating",
}


_IMPERIAL_PACKING = (
    ("m2_per_box", "box_area_sqft", 0.09290304),
    ("weight_per_box_kg", "box_weight_lb", 0.45359237),
    ("m2_per_pallet", "pallet_area_sqft", 0.09290304),
    ("weight_per_pallet_kg", "pallet_weight_lb", 0.45359237),
)


def _metric(value: Any, factor: float) -> Optional[float]:
    try:
        return round(float(str(value).replace(",", ".")) * factor, 3)
    except (TypeError, ValueError):
        return None


def _flat_to_nested(flat: Dict[str, Any], product_name: str) -> Dict[str, Any]:
    """Map the flat column output to the nested metadata shape used by
    Stage 4.7's merge function. Returns {} if nothing was extracted."""
    if not flat:
        return {}

    out: Dict[str, Any] = {}

    # Packaging. An imperial column fills its metric key only when the metric one is absent.
    pkg: Dict[str, Any] = {}
    for key in (
        "pieces_per_box", "m2_per_box", "weight_per_box_kg",
        "boxes_per_pallet", "m2_per_pallet", "weight_per_pallet_kg",
        "pallet_dimensions_cm",
    ):
        if key in flat and flat[key] is not None:
            pkg[key] = flat[key]
    for metric, imperial, factor in _IMPERIAL_PACKING:
        if pkg.get(metric) is None and flat.get(imperial) is not None:
            converted = _metric(flat[imperial], factor)
            if converted is not None:
                pkg[metric] = converted
    if pkg:
        out["packaging"] = pkg

    # Material properties
    mp: Dict[str, Any] = {}
    if "thickness_mm" in flat and flat["thickness_mm"] is not None:
        mp["thickness_mm"] = flat["thickness_mm"]
    elif flat.get("thickness_in") is not None:
        converted = _metric(flat["thickness_in"], 25.4)
        if converted is not None:
            mp["thickness_mm"] = converted
    # finish from bullets: MATT bullet → "matte"
    if flat.get("finish_matt") and not flat.get("finish_gloss"):
        mp["finish"] = "matte"
    elif flat.get("finish_gloss") and not flat.get("finish_matt"):
        mp["finish"] = "gloss"
    if mp:
        out["material_properties"] = mp

    # Application
    uses: List[str] = []
    if flat.get("recommended_use_shower_wall"):
        uses.append("shower_wall")
    if flat.get("recommended_use_shower_floor"):
        uses.append("shower_floor")
    if flat.get("recommended_use_floor"):
        uses.append("floor")
    if uses:
        out["application"] = {"application": uses}

    return out


# ──────────────────────────────────────────────────────────────────────────
# Tier B — Claude Opus Vision fallback (reuses existing extractor)
# ──────────────────────────────────────────────────────────────────────────

def _tier_b_opus(
    pdf_path: str,
    page_indices: List[int],
    product_name: str,
    *,
    job_id: Optional[str] = None,
    product_id: Optional[str] = None,
    document_id: Optional[str] = None,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """Fallback: delegate to the existing product_spec_vision_extractor
    but with the Claude Opus model override. Returns the same nested-metadata
    shape. Only runs when Tier A coverage is below threshold.

    `job_id` + `product_id` are forwarded so the Anthropic spend gets
    attributed to the right product in `ai_usage_logs`.
    """
    # Import locally to avoid a cyclic/test-time dependency
    from app.services.products import product_spec_vision_extractor as psve

    # Pass the Opus override PER-CALL rather than mutating the module global
    # psve.CLAUDE_VISION_MODEL. The previous swap-and-restore raced across the
    # (up to 5) concurrent jobs per workspace: one job's `finally` could restore
    # the default Haiku while another was mid-flight, silently sending spec-vision
    # pages to the wrong model (S4-1). A per-call arg has no shared state.
    try:
        raw = psve.extract_specs_from_pdf_pages(
            pdf_path=pdf_path,
            product_page_range=[idx + 1 for idx in page_indices],  # 1-indexed input
            product_name=product_name,
            job_id=job_id,
            product_id=product_id,
            document_id=document_id,
            model=OPUS_FALLBACK_MODEL,
            category=category,
        )
    except Exception as e:
        logger.warning(
            f"product_spec_extractor_v2: tier_b opus fallback failed: {e}"
        )
        return {"_tier_error": f"tier_b_failed: {str(e)[:200]}"}

    if not raw:
        return {}
    status = raw.get("_status", "ok")
    failures = raw.get("_page_failures") or []
    if status == "failed":
        return {"_tier_error": f"tier_b_all_pages_failed: {failures[:5]}"}
    mapped = psve.map_vision_specs_to_product_metadata(raw, category)
    if status == "degraded":
        mapped["_tier_degraded"] = {"tier": "claude_opus_vision", "page_failures": failures}
    return mapped


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
            target = _ICON_CATEGORY_TARGETS.get(category)
            if target is None and field_registry.is_loaded:
                resolved = field_registry.resolve_field_key(category, "tiles")
                target = resolved if resolved in _ICON_CATEGORY_TARGETS.values() else None
            if target is not None:
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
    category: Optional[str] = None,
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
    # _tier_b_opus -> psve builds its prompt synchronously from the cache.
    await prefetch(("extraction", "product_spec_vision", "image_analysis"))
    await field_registry.ensure_loaded()
    log_prefix = f"[{job_id or '-'}] spec_v2 '{product_name}'"
    source_tiers: List[str] = []
    degraded_tiers: List[Dict[str, Any]] = []

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
        # - Tier A only extracts the packing row + thickness + bullet flags
        # - Tier B uniquely provides: commercial.vision_variants (SKU
        # metadata per color), commercial.grout_color_codes (per-color
        # grout recommendations), and any per-product performance
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
            tier_b_result = _tier_b_opus(
                pdf_path,
                page_indices,
                product_name,
                job_id=job_id,
                product_id=product_id,
                document_id=document_id,
                category=category,
            )
            if tier_b_result.get("_tier_error"):
                # Every page failed: a failure, not "no specs". Stage 4 counts it.
                return {"_source_tiers": source_tiers, **tier_b_result}
            if tier_b_result.get("_tier_degraded"):
                degraded_tiers.append(tier_b_result["_tier_degraded"])
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
        if degraded_tiers:
            merged["_degraded_tiers"] = degraded_tiers
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
