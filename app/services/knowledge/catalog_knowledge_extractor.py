"""
Catalog-wide knowledge extractor.

Ceramic tile (and similar) catalogs typically have 4-8 pages of shared,
catalog-wide content at the END that applies to ALL products from that
catalog: iconography legends, technical standards, installation guides,
care/cleaning instructions, sustainability claims, legal notices.

These pages are NOT product-specific — they describe rules and guides that
cover every product in the catalog. We extract them ONCE per document and
link the resulting KB docs to every product in that document via
kb_doc_attachments.

How it works:

  1. `extract_catalog_knowledge_from_pdf(document_id, pdf_path, product_ids)`
     scans the PDF pages looking for knowledge content (heuristic: last N
     pages + text-density check + keyword match).
  2. For each matching page, Claude Haiku Vision reads the page and returns
     structured JSON describing the page type and the content in markdown.
  3. For each extracted section, create a kb_docs row with
     `metadata.auto_generated=true, metadata.catalog_knowledge=true`.
  4. For each product in the document, insert a kb_doc_attachments row
     linking the new kb_doc to that product with the appropriate
     relationship_type (regulation, installation, care, sustainability,
     certification).
  5. Voyage AI embeddings generated for semantic search.

Cost: ~5 Claude Haiku Vision calls per document (~$0.005 total). Cheap.

Complements `AutoKBDocumentService` which creates per-product docs from
metadata. This one creates catalog-wide docs from actual PDF content.
"""

import asyncio
import base64
import io
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import anthropic
import fitz  # PyMuPDF
from PIL import Image

from app.services.core.anthropic_error_reporter import report_anthropic_failure

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

KNOWLEDGE_VISION_MODEL = os.getenv(
    "CATALOG_KNOWLEDGE_VISION_MODEL",
    "claude-haiku-4-5-20251001",
)

PAGE_RENDER_DPI = 200  # lower than spec extraction — knowledge pages are text-heavy
MAX_IMAGE_BYTES = 4_500_000
MAX_PAGES_TO_SCAN = 10  # cap on knowledge pages per catalog (last 10 pages)

# Prompt returns a flat JSON schema so we can loop over fields without
# guessing nested structure.
KNOWLEDGE_PROMPT = """You are reading a page from the END of a ceramic tile catalog. These pages contain catalog-wide content that applies to ALL products in the catalog.

Classify this page into ONE of these types (or "none" if it's something else):

- "iconography"     — legend explaining what spec icons mean (R9/R10/R11, PEI I-V, water absorption classes, shade variation V1-V4)
- "regulation"      — technical standards, regulations, test norms (EN 14411, ANSI A137.1, DIN 51130, UNE, ISO)
- "installation"    — handling and installation recommendations (thin-set, joint width, substrate, cutting)
- "care"            — cleaning and maintenance instructions (neutral pH, sealants, stain removal)
- "sustainability"  — environmental/sustainability commitments, LEED, eco-friendly claims
- "certification"   — ISO / CE / quality certifications
- "legal"           — copyright, legal notices, trademarks
- "brand"           — brand introduction, mission statement, about the company
- "none"            — cover page, product photos, index, contact info, or anything not matching the above

Return STRICT JSON ONLY (no prose, no markdown fences):

{
  "page_type": "iconography" | "regulation" | "installation" | "care" | "sustainability" | "certification" | "legal" | "brand" | "none",
  "title": "Concise section title, e.g. 'Technical Standards' or 'Care Instructions'",
  "content_markdown": "The page content as clean markdown. Preserve structure (headings, lists, tables). Strip page numbers, artifacts. If text is bilingual, keep ONLY the English version. Maximum 3000 characters.",
  "key_points": ["bullet 1", "bullet 2", "..."],
  "certifications": ["ISO 14001", "ISO 9001", "CE", "LEED", "EN 14411", "..."],
  "language": "en"
}

When page_type is "certification", "regulation", or "sustainability", ALSO extract every certification, standard, or compliance mark visible on the page into the "certifications" array. Include ISO numbers, CE marks, EN/ANSI/DIN standards, LEED credits, sustainability marks, and quality badges. Use [] when none are visible.

If page_type is "none", return {"page_type": "none", "title": null, "content_markdown": null, "key_points": [], "certifications": [], "language": null}.

No prose. No markdown fences. JSON only."""


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

def _render_pdf_page_to_png(pdf_path: str, page_index: int, dpi: int = PAGE_RENDER_DPI) -> bytes:
    doc = fitz.open(pdf_path)
    try:
        return doc[page_index].get_pixmap(dpi=dpi).tobytes("png")
    finally:
        doc.close()


def _shrink_png_if_needed(png_bytes: bytes, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    if len(png_bytes) <= max_bytes:
        return png_bytes
    im = Image.open(io.BytesIO(png_bytes))
    for edge in (2000, 1600, 1200, 900):
        scaled = im.copy()
        scaled.thumbnail((edge, edge))
        buf = io.BytesIO()
        scaled.save(buf, format="PNG", optimize=True)
        out = buf.getvalue()
        if len(out) <= max_bytes:
            return out
    return out


def _get_source_pdf_path(document_id: str) -> Optional[str]:
    for p in [
        f"/tmp/pdf_processor_{document_id}/{document_id}.pdf",
        f"/tmp/pdf_processor_{document_id}/source.pdf",
    ]:
        if os.path.exists(p):
            return p
    return None


def _pages_to_scan(pdf_path: str) -> List[int]:
    """Pick the candidate 'knowledge' pages.

    Strategy: catalog-wide knowledge content typically lives in the last
    6-10 pages (after all products are listed). We scan the LAST N pages
    (default 10) — Claude will classify them and non-knowledge pages get
    filtered by `page_type == 'none'`.
    """
    doc = fitz.open(pdf_path)
    total = doc.page_count
    doc.close()

    scan_count = min(MAX_PAGES_TO_SCAN, max(1, total // 4))
    start = max(0, total - scan_count)
    return list(range(start, total))


def _call_claude_vision_knowledge(png_bytes: bytes) -> Optional[Dict[str, Any]]:
    if not ANTHROPIC_API_KEY:
        logger.error("catalog_knowledge_extractor: ANTHROPIC_API_KEY not set")
        return None

    png_bytes = _shrink_png_if_needed(png_bytes)
    b64 = base64.b64encode(png_bytes).decode("utf-8")

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=KNOWLEDGE_VISION_MODEL,
            max_tokens=3000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": KNOWLEDGE_PROMPT},
                ],
            }],
        )
    except Exception as e:
        report_anthropic_failure(e, service="catalog_knowledge_extractor")
        logger.warning(f"catalog_knowledge_extractor: Claude call failed: {e}")
        return None

    text = (resp.content[0].text if resp.content else "").strip()
    if text.startswith("```"):
        inner = text.split("```", 2)
        if len(inner) >= 2:
            text = inner[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(
            f"catalog_knowledge_extractor: JSON parse failed ({e}); raw[:200]={text[:200]!r}"
        )
        return None


# ──────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────

# Map page_type → kb_doc_attachments.relationship_type.
#
# kb_doc_attachments.relationship_type has a CHECK constraint limiting it to:
#   primary | supplementary | related | certification | specification.
# Every other value (regulation, installation, care, etc.) gets rejected
# with 23514 at insert time. We fold the richer page_type taxonomy down
# onto that vocabulary here; the ORIGINAL page_type is still preserved in
# `kb_docs.metadata.page_type` for display purposes on the frontend.
PAGE_TYPE_TO_RELATIONSHIP: Dict[str, str] = {
    "iconography":    "related",
    "regulation":     "specification",
    "installation":   "specification",
    "care":           "supplementary",
    "sustainability": "supplementary",
    "certification":  "certification",
    "legal":          "related",
    "brand":          "related",
}


async def extract_catalog_knowledge_from_pdf(
    document_id: str,
    workspace_id: str,
    pdf_path: Optional[str],
    product_ids: List[str],
    supabase: Any,
    logger_instance: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Extract catalog-wide knowledge content and create kb_docs + attachments.

    Args:
        document_id: Document identifier.
        workspace_id: Workspace identifier.
        pdf_path: Path to source PDF on disk. Falls back to _get_source_pdf_path().
        product_ids: All products from this document — each KB doc will be
                     attached to every product in this list.
        supabase: Supabase client (wrapper with .client attribute).
        logger_instance: Optional logger (defaults to module logger).

    Returns:
        Stats dict: pages_scanned, pages_with_content, docs_created,
                    attachments_created, errors.
    """
    log = logger_instance or logger
    stats = {
        "pages_scanned": 0,
        "pages_with_content": 0,
        "docs_created": 0,
        "attachments_created": 0,
        "certifications_propagated": 0,
        "errors": [],
    }

    # Aggregate every certification/standard spotted across the catalog-wide
    # knowledge pages. The catalog-level set applies to every product in the
    # document — after we've built this set we write it to each product's
    # metadata.compliance.certifications so the UI can render it as chips.
    catalog_certifications: List[str] = []
    seen_cert_norms: set = set()

    # Resolve PDF path
    if not pdf_path or not os.path.exists(pdf_path):
        pdf_path = _get_source_pdf_path(document_id)
    if not pdf_path:
        log.warning(f"catalog_knowledge_extractor: no PDF on disk for {document_id}")
        return stats

    if not product_ids:
        log.info(f"catalog_knowledge_extractor: no products for {document_id}, skipping")
        return stats

    # Find candidate pages
    try:
        page_indices = _pages_to_scan(pdf_path)
    except Exception as e:
        log.warning(f"catalog_knowledge_extractor: page scan failed: {e}")
        stats["errors"].append(f"page_scan: {e}")
        return stats

    log.info(
        f"📖 catalog_knowledge_extractor: scanning {len(page_indices)} tail pages "
        f"of {pdf_path} for {len(product_ids)} products"
    )

    # Try to import embedding service (optional — embeddings are best-effort)
    emb_service = None
    try:
        from app.services.embeddings.real_embeddings_service import RealEmbeddingsService
        emb_service = RealEmbeddingsService()
    except Exception as e:
        log.warning(f"   ⚠️ RealEmbeddingsService unavailable: {e}")

    for idx in page_indices:
        stats["pages_scanned"] += 1
        try:
            png = _render_pdf_page_to_png(pdf_path, idx)
        except Exception as e:
            log.warning(f"   render page {idx} failed: {e}")
            stats["errors"].append(f"render_{idx}: {e}")
            continue

        data = _call_claude_vision_knowledge(png)
        if not data:
            continue

        page_type = data.get("page_type") or "none"

        # Collect certifications regardless of page_type — Claude may find a
        # CE mark or ISO badge on a legal page or brand page too.
        page_certs = data.get("certifications") or []
        if isinstance(page_certs, list):
            for c in page_certs:
                if not isinstance(c, str):
                    continue
                cert = c.strip()
                if not cert:
                    continue
                norm = cert.lower().replace(" ", "").replace("-", "")
                if norm not in seen_cert_norms:
                    seen_cert_norms.add(norm)
                    catalog_certifications.append(cert)

        if page_type == "none" or not data.get("content_markdown"):
            continue

        stats["pages_with_content"] += 1

        title = (data.get("title") or "").strip() or f"{page_type.title()} (page {idx + 1})"
        content_md = (data.get("content_markdown") or "").strip()
        key_points = data.get("key_points") or []
        relationship_type = PAGE_TYPE_TO_RELATIONSHIP.get(page_type, "related")

        # Upsert kb_doc via RPC — idempotent on (source_document_id, page_index, title).
        # No need for a pre-check: the RPC handles insert-vs-update based on the
        # natural key, and a duplicate run will simply refresh the row in place.
        try:
            rpc_result = supabase.client.rpc(
                "upsert_kb_doc",
                {
                    "p_workspace_id": workspace_id,
                    "p_title": title,
                    "p_content": content_md,
                    "p_content_markdown": content_md,
                    "p_summary": (
                        " ".join(key_points[:3])[:500] if key_points else content_md[:300]
                    ),
                    "p_status": "published",
                    "p_visibility": "workspace",
                    "p_metadata": {
                        "auto_generated": True,
                        "catalog_knowledge": True,
                        "page_type": page_type,
                        "relationship_type": relationship_type,
                        "source_document_id": document_id,
                        "source_page_index": idx,
                        "extraction_method": "claude_vision",
                        "extraction_model": KNOWLEDGE_VISION_MODEL,
                        "key_points": key_points,
                        "generated_at": datetime.utcnow().isoformat(),
                    },
                },
            ).execute()
        except Exception as e:
            log.warning(f"   ❌ kb_docs upsert failed: {e}")
            stats["errors"].append(f"upsert_{idx}: {e}")
            continue

        doc_id = rpc_result.data if isinstance(rpc_result.data, str) else None
        if not doc_id:
            continue
        stats["docs_created"] += 1

        # Generate embedding (best-effort)
        if emb_service is not None:
            try:
                emb_text = f"{title}\n\n{content_md}"
                emb_result = await emb_service.generate_all_embeddings(
                    entity_id=doc_id,
                    entity_type="kb_doc",
                    text_content=emb_text,
                )
                if emb_result.get("success"):
                    text_embedding = emb_result.get("embeddings", {}).get("text_1024")
                    if text_embedding:
                        supabase.client.table("kb_docs").update({
                            "text_embedding": text_embedding,
                            "embedding_status": "success",
                            "embedding_model": "voyage-3.5",
                            "embedding_generated_at": datetime.utcnow().isoformat(),
                        }).eq("id", doc_id).execute()
                else:
                    supabase.client.table("kb_docs").update({
                        "embedding_status": "failed",
                    }).eq("id", doc_id).execute()
            except Exception as emb_err:
                log.warning(f"   ⚠️ embedding failed for {title}: {emb_err}")

        # Attach to every product in the document
        attach_rows = [
            {
                "workspace_id": workspace_id,
                "document_id": doc_id,
                "product_id": pid,
                "relationship_type": relationship_type,
            }
            for pid in product_ids
        ]
        try:
            supabase.client.table("kb_doc_attachments").insert(attach_rows).execute()
            stats["attachments_created"] += len(attach_rows)
        except Exception as e:
            log.warning(f"   ❌ kb_doc_attachments insert failed: {e}")
            stats["errors"].append(f"attach_{idx}: {e}")

        log.info(
            f"   ✅ Created catalog KB doc '{title}' ({page_type}) "
            f"and attached to {len(product_ids)} products"
        )

    # ─── Propagate catalog certifications to every product ──────────────
    # Certifications at the catalog level (ISO / CE / LEED / EN standards)
    # apply to every product in the document. We write them to each
    # product's metadata.compliance.certifications, merging with anything
    # the product already has. Rendered as chips in the product detail UI.
    if catalog_certifications:
        log.info(
            f"📖 catalog_knowledge_extractor: propagating "
            f"{len(catalog_certifications)} certifications to "
            f"{len(product_ids)} products: {catalog_certifications}"
        )
        try:
            existing = supabase.client.table("products") \
                .select("id, metadata") \
                .in_("id", product_ids) \
                .execute()
            for row in (existing.data or []):
                try:
                    md = row.get("metadata") or {}
                    if not isinstance(md, dict):
                        md = {}
                    compliance = md.get("compliance") or {}
                    if not isinstance(compliance, dict):
                        compliance = {}

                    # Merge with what's already there (per-product vision may
                    # have found product-specific certifications too).
                    existing_certs = compliance.get("certifications") or []
                    if not isinstance(existing_certs, list):
                        existing_certs = []
                    merged_norms = {
                        str(c).lower().replace(" ", "").replace("-", "")
                        for c in existing_certs if isinstance(c, str)
                    }
                    merged_list = list(existing_certs)
                    for cert in catalog_certifications:
                        norm = cert.lower().replace(" ", "").replace("-", "")
                        if norm not in merged_norms:
                            merged_list.append(cert)
                            merged_norms.add(norm)

                    compliance["certifications"] = merged_list
                    # Trace where these came from for debugging/backfills.
                    compliance["certifications_source"] = "catalog_knowledge"
                    md["compliance"] = compliance

                    supabase.client.table("products").update({
                        "metadata": md,
                    }).eq("id", row["id"]).execute()
                    stats["certifications_propagated"] += 1
                except Exception as e:
                    log.warning(
                        f"   ⚠️ failed to propagate certs to product {row.get('id')}: {e}"
                    )
                    stats["errors"].append(f"cert_propagate_{row.get('id')}: {e}")
        except Exception as e:
            log.warning(f"   ❌ cert propagation bulk fetch failed: {e}")
            stats["errors"].append(f"cert_propagate_bulk: {e}")

    log.info(
        f"📖 catalog_knowledge_extractor done: "
        f"scanned={stats['pages_scanned']}, content_pages={stats['pages_with_content']}, "
        f"docs={stats['docs_created']}, attachments={stats['attachments_created']}, "
        f"certs_propagated={stats['certifications_propagated']}"
    )
    return stats
