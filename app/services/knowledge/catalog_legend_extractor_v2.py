"""Catalog Legend Extractor (v2) — Layer 2 of the reusable PDF spec pipeline.

------------
- Layer 1 MUST have run first. If `documents.metadata.catalog_layout`
  is missing, this service calls `analyze_catalog_layout` on-demand
  (degraded but still correct).
- Model: Claude Haiku 4.5 by default (good enough for legends which
  are mostly text/logos). Override via env var
  `CATALOG_LEGEND_VISION_MODEL=claude-opus-5` for tough catalogs.
- Per-page cost: ~$0.002-0.005 Haiku. A typical catalog has 4-8 legend
  pages → ~$0.02-0.04 total.
- Idempotent: checks `documents.metadata.catalog_legends.extracted_at`,
  which is set only when every legend page succeeded.
- Wired into job progress + Sentry per the project-wide template.
"""

import base64
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import fitz  # PyMuPDF
import sentry_sdk

from app.services.core.anthropic_error_reporter import report_anthropic_failure
from app.services.utilities.prompt_registry import load_prompt

from app.utils.tenancy import assert_document_in_workspace

logger = logging.getLogger(__name__)

def _legend_model() -> str:
    return os.getenv("CATALOG_LEGEND_VISION_MODEL", "claude-haiku-4-5")

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







# Legend type -> the `prompts` row that drives it. The six prompt bodies used to be
# hardcoded above; they live in the DB now (#347 phase 3P), keyed
# prompt_type='extraction', stage='discovery'.
CATEGORY_BY_TYPE: Dict[str, str] = {
    "icons":          "legend_icons",
    "regulation":     "legend_regulations",
    "certification":  "legend_certifications",
    "installation":   "legend_installation",
    "care":           "legend_care",
    "sustainability": "legend_sustainability",
}


async def _legend_prompt(legend_type: str) -> str:
    return await load_prompt("extraction", CATEGORY_BY_TYPE[legend_type], stage="discovery")


#: Icon categories a legend may name. `certification` is what lets an icon page feed
#: catalogue-wide certifications at all.
LEGEND_ICON_CATEGORIES = [
    "slip_resistance", "pei_rating", "water_absorption_class", "fire_rating",
    "frost_resistance", "v_rating", "traffic_rating", "finish", "thickness_mm",
    "dimensions", "application", "certification", "ip_rating",
    "electrical_class", "energy_class", "other",
]

_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": _STR}
_NULLABLE_STR = {"anyOf": [_STR, {"type": "null"}]}
_NULLABLE_NUM = {"anyOf": [{"type": "number"}, {"type": "null"}]}
_CERT_LIST = {
    "type": "array",
    "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"name": _STR, "applies_globally": {"type": "boolean"}},
        "required": ["name", "applies_globally"],
    },
}

_TYPE_PROPERTIES: Dict[str, Dict[str, Any]] = {
    "icons": {
        "icons": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "category": {"type": "string", "enum": LEGEND_ICON_CATEGORIES},
                    "code": _STR,
                    "description": _STR,
                    "standard": _NULLABLE_STR,
                },
                "required": ["category", "code", "description", "standard"],
            },
        },
        "applies_globally": {"type": "boolean"},
    },
    "regulation": {
        "standards": _STR_LIST,
        "test_methods": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "characteristic": _STR,
                    "standard": _STR,
                    "unit": _NULLABLE_STR,
                    "value_or_range": _NULLABLE_STR,
                },
                "required": ["characteristic", "standard", "unit", "value_or_range"],
            },
        },
        "applies_globally": {"type": "boolean"},
    },
    "certification": {"certifications": _CERT_LIST},
    "installation": {
        "method": _NULLABLE_STR,
        "recommended_joint_width_mm": _NULLABLE_NUM,
        "adhesive_type": _NULLABLE_STR,
        "key_points": _STR_LIST,
    },
    "care": {
        "recommended_ph": _NULLABLE_STR,
        "cleaning_products": _STR_LIST,
        "key_points": _STR_LIST,
    },
    "sustainability": {
        "commitments": _STR_LIST,
        "certifications": _CERT_LIST,
        "recycled_content_pct": _NULLABLE_NUM,
    },
}


def _legend_output_config(legend_type: str) -> Dict[str, Any]:
    properties = {"title": _STR, "content_markdown": _STR, **_TYPE_PROPERTIES[legend_type]}
    return {
        "format": {
            "type": "json_schema",
            "name": f"legend_{legend_type}",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": properties,
                "required": list(properties),
            },
        }
    }


def _global_certifications(legend_type: str, result: Dict[str, Any]) -> List[str]:
    """Certifications a legend page says apply to EVERY product in the catalogue."""
    out: List[str] = []
    for cert in result.get("certifications") or []:
        if isinstance(cert, dict) and cert.get("applies_globally") is True and cert.get("name"):
            out.append(str(cert["name"]))
    if legend_type == "icons" and result.get("applies_globally") is True:
        for icon in result.get("icons") or []:
            if isinstance(icon, dict) and icon.get("category") == "certification" and icon.get("code"):
                out.append(str(icon["code"]))
    return out


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


async def _call_claude(
    image_bytes: bytes,
    prompt: str,
    legend_type: str,
    *,
    job_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """One schema-constrained vision call. Raises on any failure; never returns a guess."""
    from app.services.core.claude_tool_call import call_with_schema

    media_type = _detect_media_type(image_bytes)
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    try:
        call = await call_with_schema(
            task="catalog_legend_extraction",
            model=_legend_model(),
            max_tokens=3500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
            output_config=_legend_output_config(legend_type),
            job_id=job_id,
            workspace_id=workspace_id,
        )
    except Exception as e:
        report_anthropic_failure(e, service="catalog_legend_extractor_v2")
        raise
    return call.data


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
    """
    # The document must belong to the workspace before anything is read or written
    # (#31 M17-4). This function reads and UPDATES `documents` by bare id and then
    # propagates catalogue-wide fields to every product in it; service role means
    # nothing below would object to a document from another tenant.
    assert_document_in_workspace(supabase, document_id, workspace_id)

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

    # No Layer 1 result: classify now. A page of unknown type is never sent through
    # every legend prompt in the hope one sticks.
    if not layout.get("analyzed_at"):
        try:
            from app.services.discovery.catalog_layout_analyzer import analyze_catalog_layout
            layout = await analyze_catalog_layout(document_id, pdf_path, supabase, job_id=job_id)
            legend_pages_by_type = layout.get("legend_pages") or {}
            doc_md["catalog_layout"] = layout
        except Exception as e:
            logger.error(f"[{job_id or '-'}] catalog_legend_extractor_v2: layout analysis failed: {e}")
            sentry_sdk.capture_exception(e)
            stats["errors"].append(f"layout: {e}")
            return stats

    await _progress(10, f"processing legend pages {legend_pages_by_type}")

    # ── Extract each legend page ───────────────────────────────────────
    extracted_legends: Dict[str, Any] = {}
    all_certifications: List[str] = []
    # Defense against malformed catalog_layout (legend_pages must be a dict
    # of list[int]). If a value is None / int / str we'd hit TypeError on
    # len() and abort the entire extraction silently — instead, coerce.
    total_pages = sum(
        len(v) if isinstance(v, list) else 0
        for v in legend_pages_by_type.values()
    ) or 1
    processed = 0

    failed_pages: List[Dict[str, Any]] = []
    for legend_type, page_indices in legend_pages_by_type.items():
        if not page_indices or not isinstance(page_indices, list):
            continue
        if legend_type not in CATEGORY_BY_TYPE:
            logger.warning(f"[{job_id or '-'}] no prompt defined for legend type '{legend_type}'")
            continue
        prompt = await _legend_prompt(legend_type)

        for page_idx in page_indices:
            processed += 1
            pct = 10 + int(80 * (processed / total_pages))

            try:
                image_bytes = _render_and_shrink(pdf_path, page_idx)
                result = await _call_claude(
                    image_bytes, prompt, legend_type, job_id=job_id, workspace_id=workspace_id,
                )
            except Exception as e:
                logger.warning(f"[{job_id or '-'}] legend page {page_idx} ({legend_type}) failed: {e}")
                stats["errors"].append(f"page_{page_idx}: {e}")
                failed_pages.append({"page_index": page_idx, "legend_type": legend_type, "error": str(e)[:300]})
                continue

            extracted_legends.setdefault(legend_type, []).append({
                "source_page_index": page_idx,
                **result,
            })
            stats["legends_extracted"] += 1
            all_certifications.extend(_global_certifications(legend_type, result))
            await _progress(pct, f"page {page_idx}: extracted {legend_type}")

    all_certifications = _dedupe_norm(all_certifications)
    stats["failed_pages"] = failed_pages
    stats["certifications_propagated"] = all_certifications

    # ── Persist the structured legend values on the document row ──────
    try:
        now = datetime.utcnow().isoformat()
        doc_md["catalog_legends"] = {
            # Only a run in which every page succeeded may stop the next run.
            "extracted_at": None if failed_pages else now,
            "attempted_at": now,
            "failed_pages": failed_pages,
            "extraction_model": _legend_model(),
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
                        # #31 M17-2: catalog-derived model output must NOT be born
                        # `published`. Paired with #29 M15-1 it is a complete persistent
                        # injection path — a supplier PDF plants instructions once, they
                        # become a published KB doc, and they are replayed into every future
                        # agent turn that retrieves them. `draft` keeps it out of retrieval
                        # until somebody looks at it.
                        "p_status": "draft",
                        "p_visibility": "private",
                        "p_metadata": {
                            "auto_generated": True,
                            "source_trust": "catalog_derived",
                            "review_required": True,
                            "catalog_knowledge": True,
                            "legend_type": resolved_type,
                            "source_document_id": document_id,
                            "source_page_index": entry.get("source_page_index"),
                            "extraction_method": "claude_vision",
                            "extraction_model": _legend_model(),
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
                        # Use upsert with on_conflict so re-running with
                        # force=True (or recovering from a partial-failure
                        # run where kb_doc was upserted with the same id)
                        # doesn't 23505-fail on the already-linked rows.
                        try:
                            (
                                supabase.client.table("kb_doc_attachments")
                                .upsert(attach_rows, on_conflict="document_id,product_id")
                                .execute()
                            )
                        except Exception as e:
                            logger.warning(
                                f"[{job_id or '-'}] kb_doc_attachments upsert failed, "
                                f"falling back to insert with duplicate-tolerant errors: {e}"
                            )
                            try:
                                (
                                    supabase.client.table("kb_doc_attachments")
                                    .insert(attach_rows)
                                    .execute()
                                )
                            except Exception as e2:
                                logger.warning(
                                    f"[{job_id or '-'}] kb_doc_attachments insert failed: {e2}"
                                )
                                stats["errors"].append(f"attach_{doc_id}: {e2}")
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
