"""
Move every hardcoded prompt out of MIVAA and into the `prompts` table (#347 phase 3P).

WHY A SCRIPT
------------
The prompt text is 26 literals totalling ~50KB, several of them f-strings whose interpolations
have to become named `{placeholders}`. Hand-transcribing that into a SQL migration would be a
third copy of each prompt, frozen at one moment, with a real chance of a transcription slip in a
document nobody reads closely. This reads the literals out of the AST, so what lands in the
database is byte-identical to what the model was already being sent.

Like `seed_field_registry.py` before it, this is the MIGRATION PATH, not a fixture. Once every
call site loads from `prompt_registry`, the literals are deleted from the code and this script is
deleted with them.

Usage:  python scripts/seed_prompts_from_code.py [--dry-run]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
GLOBAL_WORKSPACE = "00000000-0000-0000-0000-000000000000"


def M(file: str, anchor: Any, prompt_type: str, category: str, name: str,
      stage: Optional[str] = None, subcategory: Optional[str] = None,
      placeholders: Optional[List[str]] = None, description: str = "") -> Dict[str, Any]:
    """One manifest row. `anchor` is the assigned constant name, or the 1-based line number."""
    return dict(file=file, anchor=anchor, prompt_type=prompt_type, category=category, name=name,
                stage=stage, subcategory=subcategory, placeholders=placeholders or [],
                description=description)


#: Every in-code prompt, with the key it will live under.
#:
#: `placeholders` names the f-string interpolations IN SOURCE ORDER. They become `{name}` in the
#: stored template and are filled by `prompt_registry.render(...)` at the call site.
MANIFEST: List[Dict[str, Any]] = [
    M("app/api/anthropic_routes.py", 168, "extraction", "anthropic_image_analysis",
      "Anthropic Route — Image Analysis", stage="image_analysis",
      placeholders=["product_groups_context"],
      description="Ad-hoc image analysis exposed on /api/anthropic."),
    M("app/api/anthropic_routes.py", 293, "extraction", "anthropic_chunk_analysis",
      "Anthropic Route — Chunk Analysis", stage="entity_creation",
      placeholders=["chunk_content"],
      description="Ad-hoc product-content analysis exposed on /api/anthropic."),
    M("app/api/pdf_processing/stage_4_products.py", 261, "classification", "product_classification",
      "Stage 4 — Product Classification", stage="entity_creation",
      placeholders=["product_name", "product_description", "existing_category", "vocab_lines"],
      description="Assigns material_category + zone_intent from the controlled vocabulary. "
                  "{vocab_lines} is rendered from material_categories.controlled_vocab."),
    M("app/services/ai_validation/document_classifier.py", 111, "classification", "document_page",
      "Document Page Classifier", stage="discovery",
      placeholders=["page_num", "has_images", "content"],
      description="PRODUCT / SPEC / MARKETING page triage."),
    M("app/services/chunking/chunk_type_classification_service.py", 133, "classification",
      "chunk_type", "Chunk Type Classifier", stage="chunking",
      placeholders=["chunk_types"],
      description="System prompt; {chunk_types} is the allowed set."),
    M("app/services/images/segmentation_service.py", "DEFAULT_SEGMENT_PROMPT", "agent",
      "segmentation", "Material Segmentation",
      description="Zone detection over a 3D render. Was a 9,119-char silent fallback."),
    M("app/services/integrations/product_identity_service.py", "_FACET_EXTRACTION_SYSTEM",
      "tool", "price_monitor_facets", "Price Monitor — Query Facets",
      description="Decomposes a product query into identity facets."),
    M("app/services/integrations/product_identity_service.py", "_CLASSIFIER_SYSTEM",
      "tool", "price_monitor_match", "Price Monitor — Page Match",
      description="Decides whether a scraped retailer page is the same product."),
    M("app/services/integrations/sentiment_analysis_service.py", 67, "extraction", "sentiment",
      "Feedback Sentiment Analysis",
      placeholders=["material_name", "rating", "feedback_text"]),
    M("app/services/knowledge/catalog_knowledge_extractor.py", "KNOWLEDGE_PROMPT",
      "extraction", "catalog_knowledge", "Catalog Knowledge Extraction", stage="discovery"),
    M("app/services/knowledge/catalog_legend_extractor_v2.py", "ICONS_PROMPT",
      "extraction", "legend_icons", "Catalog Legend — Iconography", stage="discovery"),
    M("app/services/knowledge/catalog_legend_extractor_v2.py", "REGULATION_PROMPT",
      "extraction", "legend_regulations", "Catalog Legend — Standards", stage="discovery"),
    M("app/services/knowledge/catalog_legend_extractor_v2.py", "CERTIFICATION_PROMPT",
      "extraction", "legend_certifications", "Catalog Legend — Certifications", stage="discovery"),
    M("app/services/knowledge/catalog_legend_extractor_v2.py", "INSTALLATION_PROMPT",
      "extraction", "legend_installation", "Catalog Legend — Installation", stage="discovery"),
    M("app/services/knowledge/catalog_legend_extractor_v2.py", "CARE_PROMPT",
      "extraction", "legend_care", "Catalog Legend — Care", stage="discovery"),
    M("app/services/knowledge/catalog_legend_extractor_v2.py", "SUSTAINABILITY_PROMPT",
      "extraction", "legend_sustainability", "Catalog Legend — Sustainability", stage="discovery"),
    # Its OWN category, not a subcategory of `products`. The loader only filters subcategory
    # when one is passed, so a fragment filed under products/entity_creation could be returned
    # to the main extractor asking for the same key — quietly replacing an 11,508-char
    # extraction prompt with a 266-char fragment.
    M("app/services/metadata/dynamic_metadata_extractor.py", 497, "extraction",
      "product_name_scope",
      "Product Name Scoping Rule", stage="entity_creation",
      # Named twice on purpose: the fragment repeats the product name, and `render` replaces
      # every occurrence, so the same placeholder legitimately appears more than once.
      placeholders=["product_name", "product_name"],
      description="Injected fragment that stops SKU bleed between products on one page."),
    M("app/services/metadata/dynamic_metadata_extractor.py", 984, "classification",
      "chunk_scope", "Chunk Scope Classifier", stage="chunking",
      placeholders=["chunk_content", "product_list"]),
    M("app/services/products/enhanced_material_property_extractor.py", "EXTRACTION_SYSTEM_PROMPT",
      "extraction", "material_properties_system", "Material Properties — System",
      stage="entity_creation"),
    M("app/services/products/product_creation_service.py", 1307, "extraction",
      "fast_product_classifier_system", "Fast Product Classifier — System", stage="discovery",
      placeholders=["chunk_texts"]),
    M("app/services/products/product_creation_service.py", 1358, "extraction",
      "product_deep_analysis", "Product Deep Analysis", stage="entity_creation",
      placeholders=["content", "confidence", "reasoning"]),
    M("app/services/products/product_description_writer.py", "DESCRIPTION_PROMPT",
      "generation", "product_description", "Product Description Writer"),
    M("app/services/products/product_spec_vision_extractor.py", "SPEC_PROMPT_TEMPLATE",
      "extraction", "product_spec_vision", "Product Spec — Vision Extraction",
      stage="image_analysis"),
    M("app/services/search/rag_service.py", 2350, "extraction", "rag_vision_analysis",
      "RAG — Vision Analysis", stage="image_analysis"),
    M("app/services/search/search_deduplication_service.py", 66, "search", "query_structuring",
      "Search — Query Structuring", placeholders=["query"]),
    M("app/services/search/unified_search_service.py", 1027, "search", "query_parser_system",
      "Search — Query Parser System"),
]


def _extract(path: Path, anchor: Any, placeholders: List[str]) -> str:
    """Pull the literal out of the AST, turning interpolations into {named} placeholders."""
    tree = ast.parse(path.read_text(encoding="utf-8"))

    node = None
    if isinstance(anchor, str):
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign) and n.targets and \
               getattr(n.targets[0], "id", None) == anchor:
                node = n.value
                break
    else:
        best = None
        for n in ast.walk(tree):
            if isinstance(n, (ast.Constant, ast.JoinedStr)) and getattr(n, "lineno", None) == anchor:
                if isinstance(n, ast.Constant) and not isinstance(n.value, str):
                    continue
                # Prefer the outermost node on that line: a JoinedStr contains Constants.
                if best is None or isinstance(n, ast.JoinedStr):
                    best = n
        node = best

    if node is None:
        raise SystemExit(f"{path}: could not find prompt anchored at {anchor!r}")

    if isinstance(node, ast.Constant):
        text = node.value
        if placeholders:
            raise SystemExit(f"{path}:{anchor} is a plain string but declares {placeholders}")
        return text

    parts, holes = [], 0
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        else:
            if holes >= len(placeholders):
                raise SystemExit(
                    f"{path}:{anchor} has more interpolations than declared placeholders "
                    f"({len(placeholders)}); next is {ast.unparse(value.value)!r}"
                )
            parts.append("{" + placeholders[holes] + "}")
            holes += 1
    if holes != len(placeholders):
        raise SystemExit(f"{path}:{anchor} declares {len(placeholders)} placeholders "
                         f"but the f-string has {holes}")
    return "".join(parts)


def build_rows() -> List[Dict[str, Any]]:
    rows = []
    for entry in MANIFEST:
        text = _extract(ROOT / entry["file"], entry["anchor"], entry["placeholders"])
        rows.append({
            "prompt_type": entry["prompt_type"],
            "category": entry["category"],
            "subcategory": entry["subcategory"],
            "stage": entry["stage"],
            "name": entry["name"],
            "description": entry["description"] or None,
            "prompt_text": text,
            "workspace_id": GLOBAL_WORKSPACE,
            "is_custom": False,
            "is_default": True,
            "is_active": True,
            "status": "active",
            "used_in": [entry["file"]],
            "_source": f"{entry['file']}:{entry['anchor']}",
            "_placeholders": entry["placeholders"],
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = build_rows()
    print(f"prompts: {len(rows)}  ({sum(len(r['prompt_text']) for r in rows):,} chars)")
    for r in sorted(rows, key=lambda r: -len(r["prompt_text"])):
        ph = f"  <- {r['_placeholders']}" if r["_placeholders"] else ""
        print(f"  {len(r['prompt_text']):6}  {r['prompt_type']}/{r['stage'] or '-'}/"
              f"{r['category']}{ph}")

    if args.dry_run:
        (ROOT / "prompt_seed.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
        print("\n--dry-run: wrote prompt_seed.json, no rows written")
        return 0

    url, key = os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        print("\nSUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY required to write.", file=sys.stderr)
        return 2

    from supabase import create_client

    client = create_client(url, key)
    inserted = updated = unchanged = 0
    for row in rows:
        payload = {k: v for k, v in row.items() if not k.startswith("_")}
        query = (client.table("prompts").select("id, prompt_text, version")
                 .eq("prompt_type", row["prompt_type"]).eq("category", row["category"])
                 .eq("workspace_id", GLOBAL_WORKSPACE))
        query = query.is_("stage", "null") if row["stage"] is None else query.eq("stage", row["stage"])
        query = (query.is_("subcategory", "null") if row["subcategory"] is None
                 else query.eq("subcategory", row["subcategory"]))
        existing = (query.order("version", desc=True).limit(1).execute()).data or []

        if not existing:
            client.table("prompts").insert(payload).execute()
            inserted += 1
        elif (existing[0].get("prompt_text") or "") == row["prompt_text"]:
            unchanged += 1
        else:
            # Never overwrite a prompt somebody edited — supersede it with a new version so the
            # old text stays recoverable. The loader takes the highest active version.
            payload["version"] = (existing[0].get("version") or 1) + 1
            client.table("prompts").insert(payload).execute()
            updated += 1

    print(f"\ndone: {inserted} inserted, {updated} superseded, {unchanged} already current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
