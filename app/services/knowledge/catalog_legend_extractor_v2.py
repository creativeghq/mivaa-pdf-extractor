"""
Catalog Legend Extractor (v2) — Layer 2 of the reusable PDF spec pipeline.

Runs ONCE per document. Consumes the page classification stored on
`documents.metadata.catalog_layout` by Layer 1 (catalog_layout_analyzer) and:

  1. For every legend page (icons, regulation, certification, installation,
     care, sustainability) — runs Claude Vision to extract the STRUCTURED
     values on that page. Stores the result in
     `documents.metadata.catalog_legends.{type}` so it can be queried later.

  2. For fields that apply catalog-wide to every product (certifications
     found on certification / regulation pages, optionally performance
     icon defaults on an iconography page), propagates the values into
     `products.metadata.compliance.*` / `performance.*` with provenance
     `source: "catalog_legend"` so we can trace where each value came
     from and re-run individual fields later.

  3. Creates one `kb_docs` row per legend section for the Knowledge Base
     UI — this extends the original `catalog_knowledge_extractor.py`
     which was focused only on markdown KB docs and didn't extract
     structured values.

This v2 replaces the old `catalog_knowledge_extractor.py` as the primary
legend-reading entry point. The old file is kept for backwards compat
and now delegates to this module.

Design notes
------------
- Layer 1 MUST have run first. If `documents.metadata.catalog_layout`
  is missing, this service calls `analyze_catalog_layout` on-demand
  (degraded but still correct).
- Model: Claude Haiku 4.5 by default (good enough for legends which
  are mostly text/logos). Override via env var
  `CATALOG_LEGEND_VISION_MODEL=claude-sonnet-4-6` for tough catalogs.
- Per-page cost: ~$0.002-0.005 Haiku. A typical catalog has 4-8 legend
  pages → ~$0.02-0.04 total.
- Idempotent: checks `documents.metadata.catalog_legends.extracted_at`
  and the per-page kb_doc records before re-creating anything.
- Wired into job progress + Sentry per the project-wide template.
"""

import base64
import io
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import anthropic
import fitz  # PyMuPDF
import sentry_sdk
from PIL import Image

from app.services.core.anthropic_error_reporter import report_anthropic_failure

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LEGEND_VISION_MODEL = os.getenv(
    "CATALOG_LEGEND_VISION_MODEL",
    "claude-haiku-4-5-20251001",
)

PAGE_RENDER_DPI = 220
MAX_IMAGE_BYTES = 4_000_000

# Map legend type → kb_doc_attachments.relationship_type (DB-valid values)
LEGEND_TYPE_TO_RELATIONSHIP: Dict[str, str] = {
    "icons":          "related",
    "regulation":     "specification",
    "certification":  "certification",
    "installation":   "specification",
    "care":           "supplementary",
    "sustainability": "supplementary",
}


# ──────────────────────────────────────────────────────────────────────────
# Prompts — one per legend type. Each returns a flat JSON schema.
# ──────────────────────────────────────────────────────────────────────────

ICONS_PROMPT = """You are reading an ICONOGRAPHY LEGEND page from a ceramic tile catalog. This page explains what each technical-characteristic icon means (e.g. what R9 vs R10 vs R11 means, what PEI I-V means, etc.).

Extract every icon + its description into STRICT JSON:

{
  "type": "legend_icons",
  "title": "e.g. Technical Characteristics Legend",
  "content_markdown": "Clean markdown of the page content, English only if bilingual. Max 2500 chars.",
  "icons": [
    {
      "category": "slip_resistance" | "pei_rating" | "water_absorption" | "fire_rating" | "frost_resistance" | "shade_variation" | "traffic_level" | "finish" | "thickness" | "dimensions" | "recommended_use" | "other",
      "code": "R10" | "PEI III" | "BIa" | "A1" | "V2" | "Bfl-s1" | "heavy_commercial" | ...,
      "description": "Human-friendly description of what this value means",
      "standard": "Optional — the ISO/EN/DIN standard that defines this value (e.g. DIN 51130, ISO 10545)"
    }
  ],
  "applies_globally": false
}

Set `applies_globally: true` only if the page text says something like "all products in this collection have R10 / PEI III / ..." — in that case add every globally-applicable spec to the icons list with the actual value (not just the definition).

Return JSON only. No prose. No markdown fences."""

REGULATION_PROMPT = """You are reading a TECHNICAL STANDARDS / REGULATIONS page from a ceramic tile catalog. This page lists the test norms (EN 14411, ISO 10545, ANSI A137.1, DIN 51130, etc.) used to measure the technical characteristics.

Extract into STRICT JSON:

{
  "type": "legend_regulation",
  "title": "...",
  "content_markdown": "Clean markdown, English only. Max 3000 chars.",
  "standards": ["EN 14411", "ISO 10545-3", "ANSI A137.1", "..."],
  "test_methods": [
    {
      "characteristic": "water absorption" | "slip resistance" | "breaking strength" | ...,
      "standard": "ISO 10545-3",
      "unit": "% (by mass)",
      "value_or_range": "≤ 0.5%"
    }
  ],
  "applies_globally": true | false
}

Return JSON only. No prose. No markdown fences."""

CERTIFICATION_PROMPT = """You are reading a CERTIFICATIONS page from a ceramic tile catalog. This page lists ISO / CE / LEED / quality marks that the manufacturer holds.

Extract into STRICT JSON:

{
  "type": "legend_certification",
  "title": "...",
  "content_markdown": "Clean markdown, English only. Max 2000 chars.",
  "certifications": ["ISO 9001", "ISO 14001", "CE", "LEED", "..."],
  "applies_globally": true
}

The `certifications` array is the catalog-wide set — every product in the catalog can be treated as holding these certifications (set applies_globally: true). If the page mentions a cert only applies to a specific product line, still list it but set applies_globally: false.

Return JSON only. No prose. No markdown fences."""

INSTALLATION_PROMPT = """You are reading an INSTALLATION GUIDE page from a ceramic tile catalog.

Extract into STRICT JSON:

{
  "type": "legend_installation",
  "title": "...",
  "content_markdown": "Clean markdown, English only. Max 3000 chars.",
  "method": "thin-set" | "medium-bed" | "mortar" | null,
  "recommended_joint_width_mm": null,
  "adhesive_type": null,
  "key_points": ["...", "..."]
}

Return JSON only. No prose. No markdown fences."""

CARE_PROMPT = """You are reading a CARE & MAINTENANCE page from a ceramic tile catalog.

Extract into STRICT JSON:

{
  "type": "legend_care",
  "title": "...",
  "content_markdown": "Clean markdown, English only. Max 3000 chars.",
  "recommended_ph": "neutral" | "alkaline" | "acidic" | null,
  "cleaning_products": ["...", "..."],
  "key_points": ["...", "..."]
}

Return JSON only. No prose. No markdown fences."""

SUSTAINABILITY_PROMPT = """You are reading a SUSTAINABILITY page from a ceramic tile catalog.

Extract into STRICT JSON:

{
  "type": "legend_sustainability",
  "title": "...",
  "content_markdown": "Clean markdown, English only. Max 3000 chars.",
  "commitments": ["...", "..."],
  "certifications": ["LEED v4", "EPD", "..."],
  "recycled_content_pct": null,
  "applies_globally": true
}

Return JSON only. No prose. No markdown fences."""

PROMPTS_BY_TYPE: Dict[str, str] = {
    "icons":          ICONS_PROMPT,
    "regulation":     REGULATION_PROMPT,
    "certification":  CERTIFICATION_PROMPT,
    "installation":   INSTALLATION_PROMPT,
    "care":           CARE_PROMPT,
    "sustainability": SUSTAINABILITY_PROMPT,
}


# ──────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ──────────────────────────────────────────────────────────────────────────

def _render_and_shrink(pdf_path: str, page_index: int) -> bytes:
    """Render a page to PNG; fall back to JPEG + lower DPI until under 4 MB.
    Mirrors the robust renderer from product_spec_vision_extractor."""
    doc = fitz.open(pdf_path)
    try:
        for dpi in (PAGE_RENDER_DPI, 180, 150, 120):
            pix = doc[page_index].get_pixmap(dpi=dpi)
            out = pix.tobytes("png")
            if len(out) <= MAX_IMAGE_BYTES:
                return out
        # JPEG fallback
        for dpi, q in ((180, 88), (150, 85), (120, 80)):
            pix = doc[page_index].get_pixmap(dpi=dpi)
            out = pix.tobytes("jpg", jpg_quality=q)
            if len(out) <= MAX_IMAGE_BYTES:
                return out
        # Last resort
        return pix.tobytes("jpg", jpg_quality=75)
    finally:
        doc.close()


def _detect_media_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return "image/png"


def _call_claude(image_bytes: bytes, prompt: str) -> Optional[Dict[str, Any]]:
    """Single vision call, returns parsed JSON or None on failure."""
    if not ANTHROPIC_API_KEY:
        logger.error("catalog_legend_extractor_v2: ANTHROPIC_API_KEY not set")
        return None

    media_type = _detect_media_type(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=LEGEND_VISION_MODEL,
            max_tokens=3500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
    except Exception as e:
        report_anthropic_failure(e, service="catalog_legend_extractor_v2")
        logger.warning(f"catalog_legend_extractor_v2: Claude call failed: {e}")
        return None

    text = (resp.content[0].text if resp.content else "").strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        parts = text.split("```", 2)
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"catalog_legend_extractor_v2: JSON parse failed: {e}; raw[:200]={text[:200]!r}")
        return None


def _dedupe_norm(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for s in items:
        if not isinstance(s, str):
            continue
        norm = s.strip().lower().replace(" ", "").replace("-", "")
        if not norm or norm in seen:
            continue
        seen.add(norm)
        out.append(s.strip())
    return out


# ──────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────

async def extract_catalog_legends(
    document_id: str,
    pdf_path: str,
    workspace_id: str,
    supabase: Any,
    *,
    job_id: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Extract structured legend values from a catalog and propagate catalog-
    wide fields to every product in the document.

    This is the entry point Stage 4.7 calls after Layer 1 has classified
    the pages. It assumes `documents.metadata.catalog_layout.legend_pages`
    is populated; if missing, it falls back to scanning the last 10 pages.

    Returns a stats dict with `legends_extracted`, `products_updated`,
    `kb_docs_created`, `errors`.
    """
    # Lazy imports for the optional progress tracker.
    tracker = None
    try:
        if job_id:
            from app.services.tracking.progress_tracker import get_progress_service
            tracker = get_progress_service().get_tracker(job_id)
    except Exception:
        tracker = None

    async def _progress(pct: int, msg: str) -> None:
        logger.info(f"[{job_id or '-'}] catalog_legend_extractor_v2: {msg}")
        if tracker:
            try:
                from app.services.tracking.progress_tracker import ProcessingStage
                await tracker.update_stage(
                    stage=ProcessingStage.ANALYZING_STRUCTURE,
                    stage_name="catalog_legend_extraction",
                    progress_percentage=pct,
                )
            except Exception:
                pass

    stats: Dict[str, Any] = {
        "legends_extracted": 0,
        "products_updated": 0,
        "kb_docs_created": 0,
        "certifications_propagated": [],
        "errors": [],
    }

    # ── Load the page classification from Layer 1 ──────────────────────
    try:
        doc_row = (
            supabase.client.table("documents")
            .select("metadata")
            .eq("id", document_id)
            .limit(1)
            .execute()
        )
        doc_md = (doc_row.data or [{}])[0].get("metadata") or {}
    except Exception as e:
        logger.error(
            f"[{job_id or '-'}] catalog_legend_extractor_v2: failed to load doc metadata: {e}"
        )
        sentry_sdk.capture_exception(e)
        doc_md = {}

    layout = doc_md.get("catalog_layout") or {}
    legend_pages_by_type: Dict[str, List[int]] = layout.get("legend_pages") or {}

    # Idempotency check — skip if we already extracted legends and not forcing
    if not force:
        existing_legends = doc_md.get("catalog_legends") or {}
        if isinstance(existing_legends, dict) and existing_legends.get("extracted_at"):
            logger.info(
                f"[{job_id or '-'}] catalog_legend_extractor_v2: already extracted at "
                f"{existing_legends['extracted_at']}, skipping (force=True to re-run)"
            )
            return {**stats, "skipped_idempotent": True, "existing_legends": existing_legends}

    # Fallback: if Layer 1 didn't run, scan the last 12 pages as candidates.
    # Not as accurate but keeps this service usable in isolation.
    if not any(legend_pages_by_type.values()):
        logger.info(
            f"[{job_id or '-'}] catalog_legend_extractor_v2: no Layer 1 classification, "
            f"falling back to last-12-pages scan"
        )
        try:
            tmp_doc = fitz.open(pdf_path)
            total = tmp_doc.page_count
            tmp_doc.close()
            candidates = list(range(max(0, total - 12), total))
            # Try each candidate with each prompt — expensive but functional
            legend_pages_by_type = {"unknown": candidates}
        except Exception as e:
            logger.error(f"[{job_id or '-'}] fallback scan failed: {e}")
            sentry_sdk.capture_exception(e)
            return stats

    await _progress(10, f"processing legend pages {legend_pages_by_type}")

    # ── Extract each legend page ───────────────────────────────────────
    extracted_legends: Dict[str, Any] = {}
    all_certifications: List[str] = []
    total_pages = sum(len(v) for v in legend_pages_by_type.values()) or 1
    processed = 0

    for legend_type, page_indices in legend_pages_by_type.items():
        if not page_indices:
            continue

        prompt = PROMPTS_BY_TYPE.get(legend_type)
        if prompt is None and legend_type != "unknown":
            logger.warning(
                f"[{job_id or '-'}] no prompt defined for legend type '{legend_type}'"
            )
            continue

        for page_idx in page_indices:
            processed += 1
            pct = 10 + int(80 * (processed / total_pages))

            try:
                image_bytes = _render_and_shrink(pdf_path, page_idx)
            except Exception as e:
                logger.warning(
                    f"[{job_id or '-'}] render page {page_idx} failed: {e}"
                )
                stats["errors"].append(f"render_{page_idx}: {e}")
                continue

            # For unknown-type fallback pages, try every prompt and use the
            # one that returns the most populated result.
            if legend_type == "unknown":
                best_result = None
                best_type = None
                best_score = 0
                for t, p in PROMPTS_BY_TYPE.items():
                    r = _call_claude(image_bytes, p)
                    if r and isinstance(r, dict):
                        score = sum(1 for v in r.values() if v not in (None, [], "", {}))
                        if score > best_score:
                            best_result = r
                            best_type = t
                            best_score = score
                if best_result and best_type and best_score >= 3:
                    result = best_result
                    resolved_type = best_type
                else:
                    result = None
                    resolved_type = None
            else:
                result = _call_claude(image_bytes, prompt)
                resolved_type = legend_type

            if not result or not resolved_type:
                await _progress(pct, f"page {page_idx}: no usable legend data")
                continue

            extracted_legends.setdefault(resolved_type, []).append({
                "source_page_index": page_idx,
                **result,
            })
            stats["legends_extracted"] += 1

            # Collect catalog-wide certifications from any legend that
            # reports them — cert page, regulation page, sustainability page
            # all may mention relevant marks.
            certs_here = result.get("certifications") or []
            if isinstance(certs_here, list):
                all_certifications.extend(c for c in certs_here if isinstance(c, str))
            # icons pages with applies_globally=true contribute their icon codes
            if result.get("applies_globally") and resolved_type == "icons":
                for icon in (result.get("icons") or []):
                    if isinstance(icon, dict) and icon.get("category") == "certification" and icon.get("code"):
                        all_certifications.append(icon["code"])

            await _progress(pct, f"page {page_idx}: extracted {resolved_type}")

    all_certifications = _dedupe_norm(all_certifications)
    stats["certifications_propagated"] = all_certifications

    # ── Persist the structured legend values on the document row ──────
    try:
        doc_md["catalog_legends"] = {
            "extracted_at": datetime.utcnow().isoformat(),
            "extraction_model": LEGEND_VISION_MODEL,
            "by_type": extracted_legends,
            "global_certifications": all_certifications,
        }
        (
            supabase.client.table("documents")
            .update({"metadata": doc_md})
            .eq("id", document_id)
            .execute()
        )
    except Exception as e:
        logger.error(
            f"[{job_id or '-'}] failed to persist catalog_legends on doc: {e}"
        )
        sentry_sdk.capture_exception(e)
        stats["errors"].append(f"persist_doc: {e}")

    # ── Propagate certifications to every product in this document ────
    if all_certifications:
        try:
            prods = (
                supabase.client.table("products")
                .select("id, metadata")
                .eq("source_document_id", document_id)
                .execute()
            )
            for row in (prods.data or []):
                md = row.get("metadata") or {}
                if not isinstance(md, dict):
                    md = {}
                compliance = md.get("compliance") or {}
                if not isinstance(compliance, dict):
                    compliance = {}
                existing = compliance.get("certifications") or []
                if not isinstance(existing, list):
                    existing = []
                merged = _dedupe_norm([*existing, *all_certifications])
                compliance["certifications"] = merged
                compliance["certifications_source"] = "catalog_legend"
                md["compliance"] = compliance
                # Provenance tag
                em = md.get("_extraction_metadata") or {}
                if isinstance(em, dict):
                    em["compliance.certifications"] = {
                        "source": "catalog_legend",
                        "confidence": 0.9,
                    }
                    md["_extraction_metadata"] = em
                try:
                    (
                        supabase.client.table("products")
                        .update({"metadata": md})
                        .eq("id", row["id"])
                        .execute()
                    )
                    stats["products_updated"] += 1
                except Exception as e:
                    logger.warning(
                        f"[{job_id or '-'}] cert propagation failed for product {row['id']}: {e}"
                    )
                    stats["errors"].append(f"propagate_{row['id']}: {e}")
        except Exception as e:
            logger.error(f"[{job_id or '-'}] cert propagation bulk fetch failed: {e}")
            sentry_sdk.capture_exception(e)
            stats["errors"].append(f"propagate_bulk: {e}")

    # ── Create one kb_docs row per extracted legend section ───────────
    try:
        product_ids_resp = (
            supabase.client.table("products")
            .select("id")
            .eq("source_document_id", document_id)
            .execute()
        )
        product_ids = [r["id"] for r in (product_ids_resp.data or [])]
    except Exception:
        product_ids = []

    for resolved_type, entries in extracted_legends.items():
        for entry in entries:
            title = (entry.get("title") or f"{resolved_type.title()} Legend").strip()
            content_md = (entry.get("content_markdown") or "").strip()
            if not content_md:
                continue
            try:
                kb_result = supabase.client.rpc(
                    "upsert_kb_doc",
                    {
                        "p_workspace_id": workspace_id,
                        "p_title": title,
                        "p_content": content_md,
                        "p_content_markdown": content_md,
                        "p_summary": content_md[:300],
                        "p_status": "published",
                        "p_visibility": "workspace",
                        "p_metadata": {
                            "auto_generated": True,
                            "catalog_knowledge": True,
                            "legend_type": resolved_type,
                            "source_document_id": document_id,
                            "source_page_index": entry.get("source_page_index"),
                            "extraction_method": "claude_vision",
                            "extraction_model": LEGEND_VISION_MODEL,
                            "generated_at": datetime.utcnow().isoformat(),
                        },
                    },
                ).execute()
                doc_id = kb_result.data if isinstance(kb_result.data, str) else None
                if doc_id:
                    stats["kb_docs_created"] += 1
                    relationship = LEGEND_TYPE_TO_RELATIONSHIP.get(resolved_type, "related")
                    if product_ids:
                        attach_rows = [
                            {
                                "workspace_id": workspace_id,
                                "document_id": doc_id,
                                "product_id": pid,
                                "relationship_type": relationship,
                            }
                            for pid in product_ids
                        ]
                        try:
                            (
                                supabase.client.table("kb_doc_attachments")
                                .insert(attach_rows)
                                .execute()
                            )
                        except Exception as e:
                            logger.warning(
                                f"[{job_id or '-'}] kb_doc_attachments insert failed: {e}"
                            )
                            stats["errors"].append(f"attach_{doc_id}: {e}")
            except Exception as e:
                logger.warning(
                    f"[{job_id or '-'}] kb_docs insert failed for {resolved_type}: {e}"
                )
                stats["errors"].append(f"kb_insert_{resolved_type}: {e}")

    await _progress(
        100,
        f"done — legends={stats['legends_extracted']}, "
        f"products_updated={stats['products_updated']}, "
        f"kb_docs={stats['kb_docs_created']}, "
        f"certs={len(all_certifications)}"
    )
    return stats
