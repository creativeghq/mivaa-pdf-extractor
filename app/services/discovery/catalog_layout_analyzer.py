"""
Catalog Layout Analyzer — Layer 1 of the reusable PDF spec extraction pipeline.

Runs ONCE per document, up front. Classifies every PDF page into one of the
categories below and stores the result in `documents.metadata.catalog_layout`
so every later stage can look up "which pages hold what kind of data" without
re-discovering.

Page categories
---------------
- product_spec          per-product spec / packing table (Harmony-style)
- product_photo         product photography / brochure spread
- legend_icons          iconography legend (R9/R10/R11, PEI I-V, etc.)
- legend_regulation     technical standards (EN 14411, ANSI, DIN, ISO)
- legend_certification  CE / ISO 9001 / ISO 14001 / LEED badges
- legend_installation   how-to-install guide (thin-set, joint width)
- legend_care           cleaning / maintenance guide
- legend_sustainability environmental / eco-friendly / LEED claims
- index_page            table of contents / SKU index
- bio                   designer bio / studio intro
- cover                 cover / intro spread
- other                 couldn't classify

Output shape written to `documents.metadata.catalog_layout`:
    {
      "analyzed_at": "2026-04-11T14:00:00Z",
      "total_pages": 71,
      "page_types": { "0": "cover", "1": "toc", ..., "15": "product_spec", ... },
      "legend_pages": {
        "icons":          [65, 66],
        "regulation":     [67],
        "certification":  [68],
        "installation":   [69],
        "care":           [70],
        "sustainability": [],
      },
      "product_pages_by_name": {
        "VALENOVA": [11, 13, 14, 15],
        "PIQUE":    [17, 18, 19, 20],
        ...
      },
      "signals_used": ["text_scan", "keyword_match", "name_match"],
    }

Design notes
------------
- **Zero Claude/LLM calls.** Uses only PyMuPDF's native `get_text()`. A full
  71-page catalog analyzes in ~2 seconds.
- **Reusable across catalog layouts.** The keyword lists below cover Harmony,
  Peronda, Aparici, Atlas Concorde, Marazzi, Florim, and Coem conventions.
  Add new keywords to the sets at the top of the file when a new catalog
  reveals an unseen pattern.
- **Name-aware.** Takes a list of known product names (from the Stage 0/4
  extraction) and tags every page where a product name appears as
  `product_spec` or `product_photo`.
- **Idempotent.** Checking `documents.metadata.catalog_layout.analyzed_at`
  before re-running skips work on docs that have already been analyzed.
"""

import logging
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import fitz  # PyMuPDF
import sentry_sdk

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Keyword lists — case + accent insensitive matching
# ──────────────────────────────────────────────────────────────────────────

# Strong signals that a page is the spec table for one or more products.
SPEC_TABLE_KEYWORDS = {
    "TECHNICAL CHARACTERISTICS",
    "TECHNICAL CHARACTERISTIC",
    "CARACTERISTICAS TECNICAS",
    "CARATTERISTICHE TECNICHE",
    "CARACTERISTIQUES TECHNIQUES",
    "PACKING",
    "PACKAGING",
    "EMBALAJE",
    "PCS / BOX",
    "PCS/BOX",
    "PIECES / BOX",
    "M2 / BOX",
    "BOXES / PALLET",
    "BOXES/PALLET",
    "WEIGHT / BOX",
    "WEIGHT / PALLET",
}

# Strong signals that a page is the iconography legend (explains what each
# pictogram means — R9/R10/R11, PEI I-V, BIa/BIb, V1/V2/V3/V4, A1/A2/Bfl).
LEGEND_ICONS_KEYWORDS = {
    # Slip resistance
    "SLIP RESISTANCE",
    "RESISTENCIA AL DESLIZAMIENTO",
    "SCIVOLOSITA",
    "DIN 51097",
    "DIN 51130",
    "PENDULUM",
    # PEI
    "PEI RATING",
    "ABRASION RESISTANCE",
    "RESISTENCIA A LA ABRASION",
    "PEI I",
    "PEI II",
    "PEI III",
    "PEI IV",
    "PEI V",
    # Water absorption
    "WATER ABSORPTION",
    "ABSORCION DE AGUA",
    "ASSORBIMENTO",
    "BIA",
    "BIB",
    "BIIA",
    "BIIB",
    "BIII",
    # Shade variation
    "SHADE VARIATION",
    "VARIAZIONE",
    "V1",
    "V2",
    "V3",
    "V4",
    # Frost / fire
    "FROST RESISTANCE",
    "RESISTENCIA AL HIELO",
    "FIRE RATING",
    "REACTION TO FIRE",
    "BFL-S1",
    # Traffic
    "TRAFFIC LEVEL",
    "FOOT TRAFFIC",
    "TRAFICO",
}

LEGEND_REGULATION_KEYWORDS = {
    "EN 14411",
    "EN-14411",
    "EN14411",
    "ISO 10545",
    "ISO-10545",
    "ANSI A137.1",
    "ANSI A137",
    "DIN 51097",
    "DIN 51130",
    "UNE-EN",
    "UNE EN",
    "TECHNICAL STANDARDS",
    "NORMATIVE",
    "NORMAS TECNICAS",
    "NORMATIVA",
    "TEST METHODS",
    "TEST NORMS",
    "METODOS DE ENSAYO",
}

LEGEND_CERTIFICATION_KEYWORDS = {
    "CERTIFICATIONS",
    "CERTIFICATES",
    "CERTIFICATION",
    "CERTIFICADO",
    "CERTIFICAZIONI",
    "ISO 9001",
    "ISO 14001",
    "ISO 45001",
    "CE MARK",
    "CE MARKING",
    "LEED",
    "EPD",
    "EMAS",
    "ECOLABEL",
    "QUALITY MANAGEMENT",
    "ENVIRONMENTAL MANAGEMENT",
}

LEGEND_INSTALLATION_KEYWORDS = {
    "INSTALLATION RECOMMENDATIONS",
    "INSTALLATION GUIDE",
    "RECOMENDACIONES DE INSTALACION",
    "INSTRUCCIONES DE INSTALACION",
    "INSTALLATION METHOD",
    "THIN-SET",
    "THIN SET",
    "ADHESIVE",
    "JOINT WIDTH",
    "SUBSTRATE",
    "CUTTING",
    "DRILLING",
    "EXPANSION JOINT",
    "GROUT",
    "CEMENT BOARD",
}

LEGEND_CARE_KEYWORDS = {
    "CARE INSTRUCTIONS",
    "CARE AND MAINTENANCE",
    "CLEANING INSTRUCTIONS",
    "CLEANING GUIDE",
    "MAINTENANCE",
    "LIMPIEZA",
    "MANUTENZIONE",
    "NEUTRAL PH",
    "STAIN REMOVAL",
    "DAILY CLEANING",
    "MANTENIMIENTO",
    "DETERGENT",
}

LEGEND_SUSTAINABILITY_KEYWORDS = {
    "SUSTAINABILITY",
    "SOSTENIBILIDAD",
    "SOSTENIBILITA",
    "ENVIRONMENTAL COMMITMENT",
    "ECO-FRIENDLY",
    "ECO FRIENDLY",
    "RECYCLED CONTENT",
    "CARBON FOOTPRINT",
    "LEED CREDITS",
    "CIRCULAR ECONOMY",
    "GREEN BUILDING",
    "ENVIRONMENT",
}

INDEX_PAGE_KEYWORDS = {
    "COLLECTIONS INDEX",
    "PRODUCT INDEX",
    "INDICE",
    "INDEX",
    "CONTENTS",
    "TABLE OF CONTENTS",
    "SUMMARY",
}

BIO_KEYWORDS = {
    "DESIGN STUDIO",
    "DESIGNER PROFILE",
    "DESIGNED BY",
    "STUDIO PROFILE",
    "FOUNDED IN",
    "ABOUT THE DESIGNER",
}


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    """Strip accents, uppercase, collapse whitespace — all comparisons go
    through this so "piqué" matches "PIQUE" and "caractéristiques" matches
    "CARACTERISTIQUES"."""
    if not s:
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s.upper().strip())


def _count_hits(text: str, keywords: Set[str]) -> int:
    """How many of the target keywords appear in this normalized text."""
    return sum(1 for kw in keywords if kw in text)


def _detect_packing_row(normalized_text: str) -> bool:
    """Heuristic: a packing table row typically shows 6+ numeric values in
    sequence near PACKING / BOX / PALLET labels."""
    if "PACKING" not in normalized_text and "BOX" not in normalized_text:
        return False
    # Count standalone numeric tokens (4+ digits or decimal values)
    numeric_tokens = re.findall(r"\b\d{1,4}[,.]?\d{0,4}\b", normalized_text)
    return len(numeric_tokens) >= 6


# ──────────────────────────────────────────────────────────────────────────
# Core classifier
# ──────────────────────────────────────────────────────────────────────────

def _classify_page(
    page_text: str,
    page_index: int,
    total_pages: int,
    known_product_names: Set[str],
) -> Tuple[str, List[str]]:
    """Return (page_type, list_of_matched_product_names) for a single page.

    Pure function — takes text + context, returns classification. Easy to
    unit-test without any PDF on disk.
    """
    norm = _normalize(page_text)
    if not norm:
        # Blank page
        return ("other", [])

    # 1. Find any product names on the page FIRST — we need this for both
    #    product_spec and product_photo classification below.
    matched_names: List[str] = []
    for name in known_product_names:
        n = _normalize(name)
        if n and n in norm:
            matched_names.append(name)

    # 2. Strong legend signals — check most-specific first
    # A page that explains R9/R10/R11/R12 or PEI I-V is an icon legend.
    icon_hits = _count_hits(norm, LEGEND_ICONS_KEYWORDS)
    cert_hits = _count_hits(norm, LEGEND_CERTIFICATION_KEYWORDS)
    reg_hits = _count_hits(norm, LEGEND_REGULATION_KEYWORDS)
    install_hits = _count_hits(norm, LEGEND_INSTALLATION_KEYWORDS)
    care_hits = _count_hits(norm, LEGEND_CARE_KEYWORDS)
    sus_hits = _count_hits(norm, LEGEND_SUSTAINABILITY_KEYWORDS)
    spec_hits = _count_hits(norm, SPEC_TABLE_KEYWORDS)

    # A page is a product_spec if it has a spec-table header AND at least
    # one known product name on it. Without a product name we treat it as
    # a legend_icons page (shared reference) even if it has spec keywords.
    if spec_hits >= 1 and matched_names:
        return ("product_spec", matched_names)

    # Icon legend needs ≥2 icon-specific keywords to avoid triggering on
    # any page that happens to mention "PEI" in passing.
    if icon_hits >= 2 and not matched_names:
        return ("legend_icons", [])

    if cert_hits >= 2:
        return ("legend_certification", matched_names)

    if reg_hits >= 2:
        return ("legend_regulation", matched_names)

    if install_hits >= 2:
        return ("legend_installation", matched_names)

    if care_hits >= 2:
        return ("legend_care", matched_names)

    if sus_hits >= 2:
        return ("legend_sustainability", matched_names)

    # 3. Index / bio / spec fallbacks
    if _count_hits(norm, INDEX_PAGE_KEYWORDS) >= 1:
        return ("index_page", [])

    if _count_hits(norm, BIO_KEYWORDS) >= 1 and not matched_names:
        return ("bio", [])

    # 4. Product photo: known product name present but no spec keywords
    if matched_names:
        return ("product_photo", matched_names)

    # 5. Covers / end pages by position + text density
    words = len(norm.split())
    if page_index < 2 and words < 50:
        return ("cover", [])
    if page_index >= total_pages - 2 and words < 50:
        return ("cover", [])

    return ("other", [])


# ──────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────

async def analyze_catalog_layout(
    document_id: str,
    pdf_path: str,
    supabase: Any,
    *,
    job_id: Optional[str] = None,
    known_product_names: Optional[List[str]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Classify every page of a PDF and persist the result to
    `documents.metadata.catalog_layout`.

    Args:
        document_id: Document UUID.
        pdf_path: Absolute path to the source PDF on disk.
        supabase: Supabase client wrapper (with `.client` attribute).
        job_id: Optional — when provided we emit progress updates via the
                ProgressTrackingService so the frontend can render a
                "analyzing catalog layout" stage in real time.
        known_product_names: Optional list of product names extracted by
                             Stage 0/4 earlier in the pipeline. When
                             provided, pages containing these names get
                             classified as product_spec / product_photo
                             and added to the `product_pages_by_name`
                             reverse-index.
        force: Skip the idempotency check and re-analyze even if results
               already exist on `documents.metadata.catalog_layout`.

    Returns:
        The `catalog_layout` dict that was written (same shape as stored).
    """
    # Lazy import so this module stays importable from unit tests that
    # don't touch the full progress-tracking stack.
    tracker = None
    try:
        if job_id:
            from app.services.tracking.progress_tracker import get_progress_service
            tracker = get_progress_service().get_tracker(job_id)
    except Exception:
        tracker = None

    async def _progress(pct: int, msg: str) -> None:
        logger.info(f"[{job_id or '-'}] catalog_layout_analyzer: {msg}")
        if tracker:
            try:
                from app.services.tracking.progress_tracker import ProcessingStage
                await tracker.update_stage(
                    stage=ProcessingStage.ANALYZING_STRUCTURE,
                    stage_name="catalog_layout_analysis",
                    progress_percentage=pct,
                )
            except Exception:
                pass

    # ── Idempotency check ──────────────────────────────────────────────
    if not force:
        try:
            existing = (
                supabase.client.table("documents")
                .select("metadata")
                .eq("id", document_id)
                .limit(1)
                .execute()
            )
            if existing.data:
                md = existing.data[0].get("metadata") or {}
                layout = md.get("catalog_layout")
                if isinstance(layout, dict) and layout.get("analyzed_at"):
                    logger.info(
                        f"[{job_id or '-'}] catalog_layout_analyzer: "
                        f"document {document_id} already analyzed at {layout['analyzed_at']}, "
                        f"skipping (use force=True to re-run)"
                    )
                    return layout
        except Exception as e:
            logger.warning(
                f"[{job_id or '-'}] catalog_layout_analyzer: idempotency check failed: {e}"
            )

    await _progress(5, f"opening PDF at {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(
            f"[{job_id or '-'}] catalog_layout_analyzer: fitz.open failed: {e}",
            exc_info=True,
        )
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("job_id", job_id)
            scope.set_tag("document_id", document_id)
            scope.set_tag("stage", "catalog_layout_analysis")
        sentry_sdk.capture_exception(e)
        raise

    total_pages = doc.page_count

    # If the caller didn't pass product names, try to load them from the
    # products table so Layer 1 still has name-matching signal.
    name_set: Set[str] = set()
    if known_product_names:
        name_set = {n for n in known_product_names if n}
    else:
        try:
            resp = (
                supabase.client.table("products")
                .select("name")
                .eq("source_document_id", document_id)
                .execute()
            )
            name_set = {r["name"] for r in (resp.data or []) if r.get("name")}
        except Exception as e:
            logger.warning(
                f"[{job_id or '-'}] catalog_layout_analyzer: could not load "
                f"product names from DB: {e}"
            )

    await _progress(
        15,
        f"classifying {total_pages} pages against {len(name_set)} product names",
    )

    page_types: Dict[str, str] = {}
    legend_pages: Dict[str, List[int]] = {
        "icons": [],
        "regulation": [],
        "certification": [],
        "installation": [],
        "care": [],
        "sustainability": [],
    }
    product_pages_by_name: Dict[str, List[int]] = {}

    try:
        for idx in range(total_pages):
            text = doc[idx].get_text() or ""
            ptype, matched_names = _classify_page(
                page_text=text,
                page_index=idx,
                total_pages=total_pages,
                known_product_names=name_set,
            )
            page_types[str(idx)] = ptype

            # Collect legend pages by sub-type
            if ptype == "legend_icons":
                legend_pages["icons"].append(idx)
            elif ptype == "legend_regulation":
                legend_pages["regulation"].append(idx)
            elif ptype == "legend_certification":
                legend_pages["certification"].append(idx)
            elif ptype == "legend_installation":
                legend_pages["installation"].append(idx)
            elif ptype == "legend_care":
                legend_pages["care"].append(idx)
            elif ptype == "legend_sustainability":
                legend_pages["sustainability"].append(idx)

            # Reverse index: product name → pages
            for name in matched_names:
                product_pages_by_name.setdefault(name, []).append(idx)

            if idx > 0 and idx % 10 == 0:
                pct = 15 + int(65 * (idx / max(1, total_pages)))
                await _progress(pct, f"classified {idx}/{total_pages} pages")
    finally:
        doc.close()

    layout = {
        "analyzed_at": datetime.utcnow().isoformat(),
        "total_pages": total_pages,
        "page_types": page_types,
        "legend_pages": legend_pages,
        "product_pages_by_name": product_pages_by_name,
        "signals_used": ["text_scan", "keyword_match", "name_match"],
        "stats": {
            "product_spec_pages": sum(1 for v in page_types.values() if v == "product_spec"),
            "product_photo_pages": sum(1 for v in page_types.values() if v == "product_photo"),
            "legend_pages": sum(len(v) for v in legend_pages.values()),
            "named_products_detected": len(product_pages_by_name),
        },
    }

    await _progress(90, "persisting catalog_layout to documents.metadata")

    # Persist: merge into existing metadata rather than overwriting.
    try:
        row = (
            supabase.client.table("documents")
            .select("metadata")
            .eq("id", document_id)
            .limit(1)
            .execute()
        )
        current_md = (row.data or [{}])[0].get("metadata") or {}
        if not isinstance(current_md, dict):
            current_md = {}
        current_md["catalog_layout"] = layout
        (
            supabase.client.table("documents")
            .update({"metadata": current_md})
            .eq("id", document_id)
            .execute()
        )
    except Exception as e:
        logger.error(
            f"[{job_id or '-'}] catalog_layout_analyzer: failed to persist: {e}",
            exc_info=True,
        )
        with sentry_sdk.configure_scope() as scope:
            scope.set_tag("job_id", job_id)
            scope.set_tag("document_id", document_id)
            scope.set_tag("stage", "catalog_layout_analysis_persist")
        sentry_sdk.capture_exception(e)
        raise

    await _progress(
        100,
        f"done — spec={layout['stats']['product_spec_pages']}, "
        f"photo={layout['stats']['product_photo_pages']}, "
        f"legend={layout['stats']['legend_pages']}, "
        f"named={layout['stats']['named_products_detected']}",
    )
    return layout
